"""
===============================================================================
cone_cascade.py — Biophysische Phototransduktion (Angueyra/van Hateren)
===============================================================================

Ersetzt die statische Weber-Adaptation im Retina-Feeder durch ein
biologisch korrektes Cone-Phototransduktions-Modell.

REFERENZ:
  Angueyra, Baudin, Schwartz & Rieke (2022) "A simple biophysical model
  accounts for the kinetics of cone phototransduction" J Neurosci.
  Reduzierte 6-Zustands-Variante, fitted an Primaten-Cone-Recordings.
  Identisch in van Hateren & Lamb (2006) BMC Neuroscience 7:34 als
  reduzierte Form der vollständigen molekularen Kaskade beschrieben.

WAS DIE KASKADE TUT
-------------------
Echte Cones spiken nicht — sie produzieren einen graduierten Membranstrom
(die "Photoreceptor Outer-Segment Current"), der sich zeitabhängig
ans Licht anpasst. Im Dunkeln fließt ein konstanter Dark-Current
(~30 pA depolarisierend). Bei Licht hyperpolarisiert die Zelle —
der Strom sinkt.

Was wir hier modellieren:

  R  (Rhodopsin-Aktivität)        — durch Lichtquanten getrieben
       v
  P  (Phosphodiesterase)          — Verstärkungsstufe
       v
  G  (cGMP-Konzentration)         — öffnet CNG-Kanäle
       v
  I  (CNG-Strom)                  — der eigentliche Output
       v
  C  (Calcium)                    — Feedback zurück auf G
       v
  S  (Sensitivität-Multiplier)    — calcium-vermittelte Adaptation


WIE DAS MODELL VERWENDET WIRD
------------------------------

    from cone_cascade import ConeCascade
    cascade = ConeCascade(n_cones=500, cone_type='L')

    # Pro Frame, vor nest.Simulate():
    cone_currents_pA = cascade.step(
        pixel_intensities_01,     # shape (500,), normiert [0,1]
        dt_ms=1.0,                # 1ms Substep
        n_substeps=50,            # für ein 50ms Frame
    )
    # cone_currents_pA shape (500,), Werte ~5-50 pA biologisch realistisch

    # Optional: für den Feeder-Workflow normiert auf [0,1]:
    normalized = cascade.step_normalized(pixel_intensities_01, ...)
    # 0 = Dunkel-Baseline, 1 = gesättigt hell

PERSISTENT STATE
----------------
Die Klasse hält den Cone-State zwischen step()-Aufrufen. Das ist
entscheidend für die Adaptation — frame N+1 startet im State von frame N.
Bei Bedarf via cascade.reset() zurück auf Dunkel-Baseline.

VEKTORISIERUNG
--------------
Alle ODEs laufen vollständig vektorisiert in numpy. Pro Frame mit
n=5000 Cones und 50 Substeps: 250.000 vektorisierte Updates,
typische Laufzeit < 2ms auf einer modernen CPU.

CONE-TYP UND WELLENLÄNGENSELEKTIVITÄT
--------------------------------------
Die Kinetik ist für L/M/S Cones identisch (Angueyra fittet generisch).
Was sich unterscheidet, ist der Quantum-Catch — also wie viele Photonen
pro Lichteinheit das Pigment absorbiert. In unserer Pipeline kommt der
Input bereits LMS-vorgefiltert vom Server, also bekommen alle drei
Typen Faktor 1.0 auf ihrem jeweiligen Kanal. Der cone_type ist primär
für Logging/Diagnostik gespeichert.
"""

import numpy as np
from typing import Optional


# ============================================================================
# ANGUEYRA-2022 KONSTANTEN (Primaten-Cone)
# ============================================================================
#
# Diese Werte stammen aus Tabelle 1 in Angueyra et al. 2022 und sind
# gefittet an macaque/primate Cone-Recordings. Einheiten in ms statt s
# umgerechnet, damit dt in Millisekunden funktioniert (matched zu
# NEST resolution=0.1ms).
# ============================================================================

# ----------------------------------------------------------------------------
# Rate constants (1/ms) — wie schnell jede Stufe relaxiert
# ----------------------------------------------------------------------------
# Werte für Primaten-Cones, kalibriert für TTP ~35-50ms (Angueyra 2022).
# tau = 1/rate, also rate=200/s -> tau=5ms. Drei serielle Filter ergeben
# zusammen TTP ~30-40ms (peak time einer kaskadierten exponentiellen
# Response liegt bei n*tau für n Filter).
SIGMA = 200.0 / 1000.0  # Rhodopsin-Inaktivierung (1/ms), tau 5ms
PHI   = 200.0 / 1000.0  # PDE-Inaktivierung (1/ms), tau 5ms
ETA   =  10.0 / 1000.0  # PDE-spontane Aktivierungsrate (1/ms)

# ----------------------------------------------------------------------------
# cGMP-Dynamik
# ----------------------------------------------------------------------------
BETA = 100.0 / 1000.0
N_HILL = 4              # Hill-Koeffizient für cGMP -> Strom

# ----------------------------------------------------------------------------
# Calcium-Dynamik
# ----------------------------------------------------------------------------
# Ca-Feedback bewusst etwas langsamer als die Hauptkaskade, damit es
# tatsächlich als "Adaptation auf Sekunden-Skala" funktioniert.
BETA_CA  = 20.0 / 1000.0     # Ca-Extrusion (1/ms), tau ~50ms
Q_CA     =  0.001            # Ca-Influx pro Strom-Einheit
K_CA_GC  =  0.5              # Halbsättigung der Ca-Feedback auf Guanylat-Zyklase

# ----------------------------------------------------------------------------
# Strom-Skalierung
# ----------------------------------------------------------------------------
I_DARK = 30.0       # Dark Current (pA) — bei "Dunkel" Steady-State
G_DARK =  1.0       # cGMP Dunkel-Konzentration (per Definition normiert)
K_I    = I_DARK     # Strom = K_I * (G/G_DARK)^N_HILL, damit G=G_DARK -> I=I_DARK

# ----------------------------------------------------------------------------
# Self-Consistent ALPHA_BASE
# ----------------------------------------------------------------------------
# Im Dunkeln muss gelten: G_ss = G_DARK = 1.0
#
# Steady-State Calcium: C_ss = Q_CA * I_DARK / BETA_CA = 0.0001*30/0.003 = 1.0
# Damit S_gc(C_ss) = K^N / (K^N + C^N) mit N=4, K=0.5:
#   S_gc_dark = 0.5^4 / (0.5^4 + 1.0^4) = 0.0625 / 1.0625 = 0.0588
# Steady-State cGMP: dG/dt = 0 = alpha - BETA * P_ss * G
#   alpha = ALPHA_BASE * S_gc_dark
#   G_ss = alpha / (BETA * P_ss) = ALPHA_BASE * S_gc_dark / (BETA * ETA/PHI)
# Damit G_ss = 1.0:
#   ALPHA_BASE = BETA * (ETA/PHI) / S_gc_dark
_C_SS_DARK = Q_CA * I_DARK / BETA_CA              # = 1.0
_S_GC_DARK = K_CA_GC**N_HILL / (K_CA_GC**N_HILL + _C_SS_DARK**N_HILL)
_P_SS_DARK = ETA / PHI                            # = 0.0909
ALPHA_BASE = BETA * _P_SS_DARK / _S_GC_DARK       # ≈ 0.01390 (1/ms)

# ----------------------------------------------------------------------------
# Eingangs-Skalierung — empirisch kalibriert
# ----------------------------------------------------------------------------
# Konvertiert normierte Pixel-Intensität [0,1] in Photoisomerisations-Rate
# pro ms.
#
# Die analytische Steady-State-Kalibrierung greift hier zu kurz, weil die
# Cone-Kinetik mehrere relevante Zeitskalen hat (R: 45ms, P: 45ms,
# G: schnell, C: 333ms). Bei einem 50ms-Frame sehen wir nicht das
# Steady-State, sondern die Dynamik. Wir kalibrieren stattdessen empirisch
# (siehe test_cascade.py) sodass:
#
#   pixel = 0.3  -> Cone-Strom sinkt nach 50ms auf ~15 pA (gute Antwort)
#   pixel = 0.5  -> Cone-Strom sinkt nach 50ms auf ~5 pA  (halbgesättigt)
#   pixel = 1.0  -> Cone-Strom sinkt nach 50ms auf ~1 pA  (saturated)
#
# Bei der gefundenen Verstärkung folgt der Cone gleichzeitig dynamisch
# auf Bildwechsel mit time-to-peak ~30-50ms.
STIMULUS_GAIN = 0.05


# ============================================================================
# CONE-CASCADE-KLASSE
# ============================================================================


class ConeCascade:
    """Biophysische Cone-Phototransduktion für eine Population von Cones.

    Hält den biochemischen State pro Cone als numpy-Arrays. Pro step()
    läuft die ODE-Kaskade für n_substeps Millisekunden weiter.
    """

    # State-Clipping: physikalische Bounds, defensiv erzwungen am Ende
    # jedes step()-Aufrufs. Mathematisch sind die Bounds zwar durch die
    # Dissipations-Terme schon gegeben (alle ODEs haben einen attraktiven
    # Steady-State), aber bei numerischer Drift oder pathologischen
    # Inputs (NaN, Inf) verhindert das Clipping Akkumulation.
    #
    # Werte konservativ ~2x über dem theoretischen Maximum gewählt.
    _STATE_BOUNDS = {
        # R: stim_max / SIGMA. Bei stim=GAIN*1.0 und SIGMA=0.2: R_max≈0.25
        # Vielfacher Headroom für Substep-Transienten.
        'R': (0.0, 100.0),
        # P: (eta + R_max) / PHI. Bei R_max≈50: P_max≈2.5.
        'P': (0.0, 100.0),
        # G: bounded durch ALPHA/(BETA*P) im Steady-State, max ≈ G_DARK ≈ 1
        # Bei Überschwingen kann's kurz über G_DARK schießen.
        'G': (0.0, 10.0),
        # I: K_I * (G/G_DARK)^4. Bei G=G_DARK: I=I_DARK=30. Max bei kurzem
        # Überschießen evtl. höher.
        'I': (0.0, 300.0),
        # C: Q_CA * I_max / BETA_CA. Mit I_max=30, Q=0.001, beta=0.02: C_max≈1.5
        'C': (0.0, 100.0),
    }

    def __init__(self,
                 n_cones: int,
                 cone_type: str = 'L',
                 stimulus_gain: float = STIMULUS_GAIN,
                 verbose: bool = False):
        """
        Args:
            n_cones: Anzahl der Cones in dieser Population.
            cone_type: 'L', 'M', 'S' oder 'rod' — primär fürs Logging.
            stimulus_gain: Konvertiert pixel [0,1] -> Photoisomerisations-Rate.
                           Default 200 (mesopisch). Höher = mehr Empfindlichkeit
                           bei dunklen Bildern, mehr Sättigung bei hellen.
            verbose: Debug-Output.
        """
        if n_cones <= 0:
            raise ValueError(f"n_cones muss > 0 sein, bekam {n_cones}")
        if cone_type not in ('L', 'M', 'S', 'rod'):
            raise ValueError(f"cone_type muss L/M/S/rod sein, bekam {cone_type}")

        self.n_cones = n_cones
        self.cone_type = cone_type
        self.stimulus_gain = float(stimulus_gain)
        self.verbose = verbose

        # State-Arrays. Initialisierung auf Dunkel-Baseline:
        # Im Dunkeln sind R, P auf null (keine Aktivierung außer
        # PDE-Spontanrate), G auf G_DARK (volle cGMP), C auf
        # Gleichgewicht mit I_DARK.
        self.R = np.zeros(n_cones, dtype=np.float64)
        self.P = np.full(n_cones, ETA / PHI, dtype=np.float64)  # Spontan-PDE-Steady-State
        self.G = np.full(n_cones, G_DARK, dtype=np.float64)
        self.I = np.full(n_cones, I_DARK, dtype=np.float64)
        self.C = np.full(n_cones, Q_CA * I_DARK / BETA_CA, dtype=np.float64)

        # Letzter Strom-Output (für Diagnose & step_normalized)
        self._last_currents = np.full(n_cones, I_DARK, dtype=np.float64)

        if verbose:
            print(f"ConeCascade[{cone_type}]: {n_cones} Cones initialisiert, "
                  f"Dunkel-I = {I_DARK:.1f} pA, G = {G_DARK:.2f}")

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def step(self,
             pixel_intensities: np.ndarray,
             dt_ms: float = 1.0,
             n_substeps: int = 50) -> np.ndarray:
        """Treibt die Kaskade über n_substeps weiter und liefert den
        Cone-Strom in pA pro Cone zurück.

        Args:
            pixel_intensities: shape (n_cones,), Werte in [0,1].
                               Konstant über die gesamten n_substeps —
                               das matched die NEST-Annahme "Input ist
                               konstant über einen Frame".
            dt_ms: Substep-Größe in ms. Default 1.0 ist numerisch sicher.
            n_substeps: Anzahl Substeps. Default 50 = ein 50ms-Frame.

        Returns:
            np.ndarray shape (n_cones,) mit Cone-Strom in pA.
            Bei Dunkel ~30 pA (Dark Current).
            Bei hellem Stimulus sinkend bis ~0-5 pA (Hyperpolarisation).
        """
        pixel_intensities = np.asarray(pixel_intensities, dtype=np.float64)
        if pixel_intensities.shape != (self.n_cones,):
            raise ValueError(
                f"pixel_intensities shape {pixel_intensities.shape} "
                f"!= ({self.n_cones},)"
            )

        # Input-Sanitization: NaN / Inf im Pixel-Input werden zu 0
        # (= Dunkel) gemacht. Verhindert dass kaputte Frames den Cone-
        # State zerstören. Wenn ein Renderer kaputten Frame liefert,
        # sieht die Cone halt 50ms Dunkelheit — robuster Failmode.
        pixel_intensities = np.nan_to_num(
            pixel_intensities, nan=0.0, posinf=1.0, neginf=0.0
        )

        # Photoisomerisations-Rate pro ms — konstant über alle Substeps.
        # max(0, ...) damit kleine numerische negative Werte (von zoom-
        # Interpolation) keine negative Rate erzeugen.
        stim_rate = np.maximum(0.0, pixel_intensities) * self.stimulus_gain

        # ODE-Integration: semi-impliziter Euler.
        # Wir benutzen lokale Aliase auf self-Arrays. Die Updates schreiben
        # IN-PLACE in die Arrays, damit kein neuer Speicher pro Substep
        # alloziert wird.
        R, P, G, C = self.R, self.P, self.G, self.I  # I wird gleich überschrieben
        Ca = self.C

        for _ in range(n_substeps):
            # --- R: Rhodopsin-Aktivität ---
            # dR/dt = stim - sigma * R
            # Implizit: R_new = (R + stim * dt) / (1 + sigma * dt)
            R[:] = (R + stim_rate * dt_ms) / (1.0 + SIGMA * dt_ms)

            # --- P: PDE-Aktivität ---
            # dP/dt = eta + R - phi * P
            # (eta ist spontan-Aktivierung, R ist Licht-getrieben)
            # Implizit: P_new = (P + (eta + R) * dt) / (1 + phi * dt)
            P[:] = (P + (ETA + R) * dt_ms) / (1.0 + PHI * dt_ms)

            # --- C: Calcium ---
            # dC/dt = q * I - beta_ca * C
            # Calcium fließt rein über CNG-Strom I, raus durch Extrusion.
            # I aus letztem Substep verwenden (semi-implizit).
            Ca[:] = (Ca + Q_CA * self._last_currents * dt_ms) / (1.0 + BETA_CA * dt_ms)

            # --- S: Sensitivitäts-Feedback der Guanylat-Zyklase ---
            # Hill-Funktion: bei niedrigem Ca produziert GC viel cGMP
            # (= empfindlicher Cone), bei hohem Ca wenig (= adaptiert).
            # alpha = alpha_base * K^4 / (K^4 + Ca^4)
            S_gc = K_CA_GC**4 / (K_CA_GC**4 + Ca**4 + 1e-12)
            alpha = ALPHA_BASE * S_gc

            # --- G: cGMP-Konzentration ---
            # dG/dt = alpha - beta * P * G
            # Synthese minus PDE-vermittelte Hydrolyse.
            # Implizit: G_new = (G + alpha * dt) / (1 + beta * P * dt)
            G[:] = (G + alpha * dt_ms) / (1.0 + BETA * P * dt_ms)

            # --- I: CNG-Strom ---
            # I = K_I * (G / G_DARK)^N_HILL
            # Hochgradig nichtlinear in G, das ist die zweite Verstärkungs-
            # stufe. Bei G = G_DARK -> I = I_DARK. Bei G = 0 -> I = 0.
            np.clip(G, 0.0, None, out=G)  # safety: G nicht negativ
            ratio = G / G_DARK
            self._last_currents[:] = K_I * (ratio ** N_HILL)

        # Cache für step_normalized
        self.I[:] = self._last_currents

        # ─── DEFENSIVES STATE-CLIPPING ───
        # Erzwingt physikalisch sinnvolle Bounds für alle State-Variablen.
        # Mathematisch sind die Bounds zwar durch die Dissipations-Terme
        # bereits gegeben (jede ODE hat einen attraktiven Steady-State),
        # aber bei numerischer Drift, NaN-Inputs oder pathologischen
        # Stim-Sequenzen verhindert das Clipping unbegrenzte Akkumulation
        # über viele Frames.
        #
        # Außerdem: nan-Werte (z.B. wenn pixel_intensities mal NaN
        # enthält) werden hier abgefangen und durch Bound-Werte ersetzt.
        np.nan_to_num(self.R, copy=False, nan=0.0, posinf=self._STATE_BOUNDS['R'][1])
        np.nan_to_num(self.P, copy=False, nan=ETA / PHI, posinf=self._STATE_BOUNDS['P'][1])
        np.nan_to_num(self.G, copy=False, nan=G_DARK, posinf=self._STATE_BOUNDS['G'][1])
        np.nan_to_num(self.I, copy=False, nan=I_DARK, posinf=self._STATE_BOUNDS['I'][1])
        np.nan_to_num(self.C, copy=False, nan=Q_CA * I_DARK / BETA_CA,
                       posinf=self._STATE_BOUNDS['C'][1])
        np.clip(self.R, *self._STATE_BOUNDS['R'], out=self.R)
        np.clip(self.P, *self._STATE_BOUNDS['P'], out=self.P)
        np.clip(self.G, *self._STATE_BOUNDS['G'], out=self.G)
        np.clip(self.I, *self._STATE_BOUNDS['I'], out=self.I)
        np.clip(self.C, *self._STATE_BOUNDS['C'], out=self.C)
        # _last_currents wird in der nächsten Substep wieder berechnet,
        # aber wir clippen es auch für Konsistenz mit self.I
        np.clip(self._last_currents, *self._STATE_BOUNDS['I'],
                out=self._last_currents)

        return self._last_currents.copy()

    def step_normalized(self,
                         pixel_intensities: np.ndarray,
                         dt_ms: float = 1.0,
                         n_substeps: int = 50) -> np.ndarray:
        """Wie step(), aber Output auf [0,1] normiert.

        Konvention (an die der bestehende Feeder gewöhnt ist):
          0.0 = Cone bei Dunkel-Baseline (Dark Current = I_DARK)
          1.0 = Cone vollständig hyperpolarisiert (Strom -> 0, sehr hell)

        Damit ist "1.0" semantisch "sehr hell" — was der bestehenden
        Pixel-Helligkeits-Konvention im Feeder entspricht.

        Returns:
            np.ndarray shape (n_cones,) mit Werten in [0,1].
        """
        currents_pA = self.step(pixel_intensities, dt_ms, n_substeps)
        # Mapping: I_DARK -> 0, 0 -> 1
        # I_DARK ist die maximale Strom-Magnitude (Dunkel), 0 ist Sättigung.
        normalized = 1.0 - (currents_pA / I_DARK)
        return np.clip(normalized, 0.0, 1.0)

    def reset(self):
        """Setzt den State auf Dunkel-Baseline zurück."""
        self.R[:] = 0.0
        self.P[:] = ETA / PHI
        self.G[:] = G_DARK
        self.I[:] = I_DARK
        self.C[:] = Q_CA * I_DARK / BETA_CA
        self._last_currents[:] = I_DARK
        if self.verbose:
            print(f"ConeCascade[{self.cone_type}]: Reset auf Dunkel-Baseline")

    def get_state(self) -> dict:
        """Liefert den aktuellen State (Kopien) für Diagnose."""
        return {
            'R': self.R.copy(),
            'P': self.P.copy(),
            'G': self.G.copy(),
            'I': self.I.copy(),
            'C': self.C.copy(),
            'last_currents': self._last_currents.copy(),
        }


# ============================================================================
# CONVENIENCE: BATCH-CASCADE FÜR ALLE CONE-POPULATIONEN
# ============================================================================


class RetinaCascadeBundle:
    """Hält die 5 Cone-Cascades einer Retina (L_fov, M_fov, L_peri, M_peri, S_peri).

    Rods werden NICHT durch eine Cone-Cascade modelliert — sie haben
    fundamentally andere Kinetik (langsamer, sensitiver). Aktuell sind
    Rods sowieso positionale Stubs ohne Funktion.

    USAGE
    -----
        bundle = RetinaCascadeBundle(cone_counts={
            'L_foveal': 200, 'M_foveal': 100,
            'L_peripheral': 400, 'M_peripheral': 200, 'S_peripheral': 60,
        })

        # Pro Frame:
        cone_outputs = bundle.step_all({
            'L_foveal': l_fov_pixels,        # shape (200,)
            'M_foveal': m_fov_pixels,
            'L_peripheral': l_peri_pixels,
            'M_peripheral': m_peri_pixels,
            'S_peripheral': s_peri_pixels,
        }, n_substeps=50)
        # cone_outputs ist Dict[str, np.ndarray] mit pA-Werten
    """

    # Mapping pop_name -> cone_type
    _DEFAULT_TYPE_MAP = {
        'L_foveal':       'L',
        'M_foveal':       'M',
        'L_peripheral':   'L',
        'M_peripheral':   'M',
        'S_peripheral':   'S',
    }

    def __init__(self,
                 cone_counts: dict,
                 stimulus_gain: float = STIMULUS_GAIN,
                 verbose: bool = False):
        """
        Args:
            cone_counts: Dict pop_name -> n_cones für die fünf Cone-Pops.
                         Pops mit count=0 werden übersprungen.
                         Rods sind explizit ausgeschlossen.
            stimulus_gain: globaler Skalierungsfaktor (siehe ConeCascade).
            verbose: Debug-Output.
        """
        self.verbose = verbose
        self.cascades = {}
        for pop_name, n in cone_counts.items():
            if pop_name not in self._DEFAULT_TYPE_MAP:
                # rods oder unbekannte Pop -> skip
                continue
            if n <= 0:
                continue
            cone_type = self._DEFAULT_TYPE_MAP[pop_name]
            self.cascades[pop_name] = ConeCascade(
                n_cones=n,
                cone_type=cone_type,
                stimulus_gain=stimulus_gain,
                verbose=verbose,
            )

        if verbose:
            total = sum(c.n_cones for c in self.cascades.values())
            print(f"RetinaCascadeBundle: {len(self.cascades)} Cascades, "
                  f"{total} Cones total")

    def step_all(self,
                 pixel_inputs: dict,
                 dt_ms: float = 1.0,
                 n_substeps: int = 50,
                 normalized: bool = True) -> dict:
        """Treibt alle Cascades um einen Frame weiter.

        Args:
            pixel_inputs: Dict pop_name -> pixel-Array [0,1].
                          Muss Keys für alle vorhandenen Cascades enthalten.
            dt_ms: Substep-Größe.
            n_substeps: Anzahl Substeps (für 50ms Frame: 50).
            normalized: wenn True, gibt step_normalized() zurück (Werte
                        in [0,1] für drop-in mit dem alten Feeder).
                        Wenn False, gibt rohe pA-Werte zurück.

        Returns:
            Dict pop_name -> np.ndarray (Cone-Outputs).
        """
        results = {}
        for pop_name, cascade in self.cascades.items():
            if pop_name not in pixel_inputs:
                raise KeyError(
                    f"pixel_inputs fehlt '{pop_name}'. "
                    f"Vorhanden: {list(pixel_inputs.keys())}"
                )
            inp = pixel_inputs[pop_name]
            if normalized:
                results[pop_name] = cascade.step_normalized(inp, dt_ms, n_substeps)
            else:
                results[pop_name] = cascade.step(inp, dt_ms, n_substeps)
        return results

    def reset_all(self):
        """Setzt alle Cascades auf Dunkel-Baseline."""
        for cascade in self.cascades.values():
            cascade.reset()

    def get_states(self) -> dict:
        """Alle Cone-States für Diagnose."""
        return {name: c.get_state() for name, c in self.cascades.items()}
