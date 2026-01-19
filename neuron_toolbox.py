import nest
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Any
from IPython.display import Image, display
from numpy import diag, array, zeros
import random
import numpy as np
import numpy as np, heapq, random
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, asdict
import json
import copy
import random
from typing import List, Tuple, Callable, Optional, Dict, Any

SCALING_FACTOR = 1.0
k= 10.0
m= np.zeros(3)
dt = 0.01
positions=[np.zeros(3)]
gsl = 8

# Multi-Compartment Neuron Models - require special handling
MC_MODELS = {'iaf_cond_alpha_mc', 'cm_default', 'cm_main', 'iaf_cond_beta_mc'}

def get_mc_recordables(model_name: str) -> list:
    """
    Returns appropriate recordables for multi-compartment models.
    Standard models use ['V_m'], MC models use compartment-specific variables.
    """
    if model_name in MC_MODELS:
        return ['V_m_s']  # Soma membrane potential
    return ['V_m']

def get_receptor_type_for_model(model_name: str, excitatory: bool = True) -> int:
    """
    Returns appropriate receptor_type for multi-compartment models.
    For standard models returns 0 (default).
    """
    if model_name in MC_MODELS:
        return 1 if excitatory else 2  # soma_exc or soma_inh
    return 0


area2models = {
    
    "V1": ["iaf_cond_alpha","aeif_cond_alpha","iaf_cond_exp","hh_psc_alpha","iaf_cond_alpha","iaf_cond_beta_gap"],
    
    "V2": ["aeif_cond_alpha","iaf_cond_exp","aeif_cond_alpha","hh_psc_alpha","iaf_cond_alpha"],
    
    "V3": ["aeif_cond_alpha","iaf_cond_exp","aeif_cond_alpha","hh_psc_alpha","iaf_cond_alpha"],
    
    "V4": ["aeif_cond_alpha","iaf_cond_exp","hh_psc_alpha","gif_cond_exp","iaf_cond_alpha","iaf_cond_beta_gap"],
    
    "V5": ["iaf_cond_alpha","aeif_cond_alpha","hh_psc_alpha","izhikevich","iaf_cond_beta_gap"],
    
    "V6": ["iaf_cond_alpha","aeif_cond_alpha","hh_psc_alpha","iaf_psc_delta","iaf_cond_alpha","iaf_cond_beta_gap"]
    
}

def apply_transform(points, rot_theta=0.0, rot_phi=0.0, m=np.zeros(3)):
    pts = np.asarray(points, dtype=float)
    
    th = np.deg2rad(rot_theta)
    ph = np.deg2rad(rot_phi)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(th), -np.sin(th)],
                   [0, np.sin(th),  np.cos(th)]])
    
    Ry = np.array([[ np.cos(ph), 0, np.sin(ph)],
                   [          0, 1,          0],
                   [-np.sin(ph), 0, np.cos(ph)]])
    
    rotM = Ry @ Rx
    
    rotated = pts @ rotM.T
    transformed = rotated + np.asarray(m, dtype=float)
    
    return transformed

def compute_angular_similarity_weights(local_points, weight_ex, weight_in, bidirectional=True):
    angles = np.arctan2(local_points[:, 1], local_points[:, 0])
    
    delta_theta = angles[None, :] - angles[:, None]
    
    delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
    
    if bidirectional:
        similarity = np.cos(delta_theta)
    else:
        shift = np.pi / 4.0
        similarity = np.cos(delta_theta - shift)
    
    weights = np.zeros_like(similarity)
    
    mask_ex = similarity > 0
    weights[mask_ex] = similarity[mask_ex] * weight_ex
    
    mask_in = similarity < 0
    weights[mask_in] = similarity[mask_in] * weight_in
    
    return weights.flatten()

def connect_neighbors_by_index(nodes, n_neighbors, weight_ex, weight_in, delay):
    gids = np.array(nodes.get("global_id"))
    N = len(gids)
    
    m = min(n_neighbors, N - 1)
    if m < 1: return
    
    sources = []
    targets = []
    weights = []
    delays = []
    
    for i in range(N):
        for offset in range(1, m + 1):
            target_idx = (i + offset) % N
            sources.append(gids[i])
            targets.append(gids[target_idx])
            
            w = weight_ex * (1.0 - (offset-1)/m)
            weights.append(w)
            delays.append(delay)
            
        for offset in range(1, m + 1):
            target_idx = (i - offset) % N
            sources.append(gids[i])
            targets.append(gids[target_idx])
            
            w = -abs(weight_in) * (1.0 - (offset-1)/m)
            weights.append(w)
            delays.append(delay)
            
    nest.Connect(sources, targets,
                 {'rule': 'one_to_one'},
                 {'synapse_model': 'static_synapse',
                  'weight': np.array(weights),
                  'delay': np.array(delays)})
    


def extract_connections(pre, post):

    conn = nest.GetConnections(pre, post)
    df = pd.DataFrame({
        "pre_gid":  conn.source,
        "post_gid": conn.target,
        "weight":   nest.GetStatus(conn, "weight"),
        "delay":    nest.GetStatus(conn, "delay")
    })
    return df


def to_adjacency(df, pre_ids, post_ids, attr="weight"):
    idx = {gid: i for i, gid in enumerate(pre_ids)}
    jdx = {gid: j for j, gid in enumerate(post_ids)}
    mat = np.zeros((len(pre_ids), len(post_ids)))
    for _, row in df.iterrows():
        i, j = idx[row.pre_gid], jdx[row.post_gid]
        mat[i, j] = row[attr]
    return mat


def plot_point_clusters(
    clusters,
    colors= None,
    marker_size = 20,
    cmap = 'tab10',
    xlabel = 'X',
    ylabel = 'Y',
    zlabel = 'Z',
    title = None,
    edgecolor='k', alpha=0.8,linewidths=1.5
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = len(clusters)
    
    if colors is None:
        cmap_obj = plt.get_cmap(cmap)
        colors = [cmap_obj(i / max(n-1, 1)) for i in range(n)]
    
    for pts, col in zip(clusters, colors):
        pts = np.asarray(pts)
        if pts.size == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=[col], s=marker_size, edgecolor=edgecolor, alpha=alpha,linewidths=linewidths)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if title:
        ax.set_title(title)
    plt.show()
    

def plot_point_clusters_normalized(
        clusters,
        colors=None,
        marker_size=20,
        cmap='tab10',
        xlabel='X', ylabel='Y', zlabel='Z',
        title=None,
        edgecolor='k', alpha=0.8, linewidths=1.5):

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    n = len(clusters)
    if colors is None:
        cmap_obj = plt.get_cmap(cmap)
        colors = [cmap_obj(i / max(n - 1, 1)) for i in range(n)]

    mins = np.full(3,  np.inf)
    maxs = np.full(3, -np.inf)

    for pts, col in zip(clusters, colors):
        pts = np.asarray(pts)
        if pts.size == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=[col], s=marker_size,
                   edgecolor=edgecolor, alpha=alpha, linewidths=linewidths)

        mins = np.minimum(mins, pts.min(axis=0))
        maxs = np.maximum(maxs, pts.max(axis=0))

    span   = max(maxs - mins)
    centre = (maxs + mins) / 2

    ax.set_xlim(centre[0] - span / 2, centre[0] + span / 2)
    ax.set_ylim(centre[1] - span / 2, centre[1] + span / 2)
    ax.set_zlim(centre[2] - span / 2, centre[2] + span / 2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

def set_axes_equal(ax):


    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d(x_middle - max_range/2, x_middle + max_range/2)
    ax.set_ylim3d(y_middle - max_range/2, y_middle + max_range/2)
    ax.set_zlim3d(z_middle - max_range/2, z_middle + max_range/2)


    
def make_div_curl(f1, f2, f3, h = 1e-5):

    def div(x, y, z):
        df1dx = (f1(x + h, y, z) - f1(x - h, y, z)) / (2*h)
        df2dy = (f2(x, y + h, z) - f2(x, y - h, z)) / (2*h)
        df3dz = (f3(x, y, z + h) - f3(x, y, z - h)) / (2*h)
        return df1dx + df2dy + df3dz

    def curl(x, y, z):
        df3dy = (f3(x, y + h, z) - f3(x, y - h, z)) / (2*h)
        df2dz = (f2(x, y, z + h) - f2(x, y, z - h)) / (2*h)
        c1 = df3dy - df2dz

        df1dz = (f1(x, y, z + h) - f1(x, y, z - h)) / (2*h)
        df3dx = (f3(x + h, y, z) - f3(x - h, y, z)) / (2*h)
        c2 = df1dz - df3dx

        df2dx = (f2(x + h, y, z) - f2(x - h, y, z)) / (2*h)
        df1dy = (f1(x, y + h, z) - f1(x, y - h, z)) / (2*h)
        c3 = df2dx - df1dy

        return np.array([c1, c2, c3])

    return div, curl


def generate_direction_similarity_matrix(
    pop1=None,
    pop2=None,
    vecfield1=(
        lambda x, y, z: 1.5 * x ** 2 + y + z,
        lambda x, y, z: 1.3 * x ** 3 - y + z,
        lambda x, y, z: 1.7 * x ** 2 + y - z,
    ),
    vecfield2=(
        lambda x, y, z: 4.7 * x - np.sin(y),
        lambda x, y, z: -1.9 * x ** 2 - y + 2,
        lambda x, y, z: 1.7 * y ** 2 + x - y,
    ),
    eps: float = 1e-12,
    plot: bool = False,
    cmap: str = "plasma",
    arrow_length: float = 0.2,
):
    if pop1 is None:
        grid = generate_cube((4, 4, 4))
        pop1 = transform_points(
            grid, [1, 2, 3], 120, 120, transform_matrix=np.eye(3), plot=plot
        )
    if pop2 is None:
        grid = generate_cube((4, 4, 4))
        pop2 = transform_points(
            grid, [3, 3, 1], 24, 77, transform_matrix=np.eye(3), plot=plot
        )

    f1v1, f2v1, f3v1 = vecfield1
    f1v2, f2v2, f3v2 = vecfield2

    vec1 = field(pop1, plot=plot, normalize=True, cmap=cmap,
                 arrow_length=arrow_length, f1=f1v1, f2=f2v1, f3=f3v1)
    vec2 = field(pop2, plot=plot, normalize=True, cmap=cmap,
                 arrow_length=arrow_length, f1=f1v2, f2=f2v2, f3=f3v2)

    u1 = vec1 / np.clip(np.linalg.norm(vec1, axis=1, keepdims=True), eps, None)
    u2 = vec2 / np.clip(np.linalg.norm(vec2, axis=1, keepdims=True), eps, None)

    return u2 @ u1.T


def generate_direction_similarity_matrix2(
    grid_shape=(4, 4, 4),
    transform1=None,
    transform2=None,
    stretch1=(1, 1, 1),
    stretch2=(1, 1, 1),
    vecfield1=(
        lambda x, y, z: 1.5 * x ** 2 + y + z,
        lambda x, y, z: 1.3 * x ** 3 - y + z,
        lambda x, y, z: 1.7 * x ** 2 + y - z,
    ),
    vecfield2=(
        lambda x, y, z: 4.7 * x - np.sin(y),
        lambda x, y, z: -1.9 * x ** 2 - y + 2,
        lambda x, y, z: 1.7 * y ** 2 + x - y,
    ),
    eps=1e-12,
    plot=False,
    cmap="plasma",
    arrow_length=0.2,
):
    if transform1 is None:
        transform1 = dict(m=[1, 2, 3], rot_theta=120, rot_phi=120)
    if transform2 is None:
        transform2 = dict(m=[3, 3, 1], rot_theta=24, rot_phi=77)

    grid = generate_cube(grid_shape)
    pop1 = transform_points(
        grid,
        transform1["m"],
        transform1["rot_theta"],
        transform1["rot_phi"],
        transform_matrix=np.diag(stretch1),
        plot=plot,
    )
    pop2 = transform_points(
        grid,
        transform2["m"],
        transform2["rot_theta"],
        transform2["rot_phi"],
        transform_matrix=np.diag(stretch2),
        plot=plot,
    )

    f1v1, f2v1, f3v1 = vecfield1
    f1v2, f2v2, f3v2 = vecfield2

    vec1 = field(pop1, plot=plot, normalize=True, cmap=cmap,
                 arrow_length=arrow_length, f1=f1v1, f2=f2v1, f3=f3v1)
    vec2 = field(pop2, plot=plot, normalize=True, cmap=cmap,
                 arrow_length=arrow_length, f1=f1v2, f2=f2v2, f3=f3v2)

    u1 = vec1 / np.clip(np.linalg.norm(vec1, axis=1, keepdims=True), eps, None)
    u2 = vec2 / np.clip(np.linalg.norm(vec2, axis=1, keepdims=True), eps, None)

    return u2 @ u1.T


class PolynomGenerator:
    def __init__(self, n=5, decay=0.5, coefficients=None):
        self.n = n
        self.terms = self.term_matrix()
        self.coefficients = self.random_coefficients() if coefficients is None else coefficients
        self.decay = decay
        self.term_power_probability = self.create_exponential_prob_vector(n=n+1, decay=decay)
    
    @property
    def coefficients(self):
        return self._coefficients
    
    @coefficients.setter
    def coefficients(self, value):
        if value is None:
            self._coefficients = self.random_coefficients()
            return

        val_arr = np.array(value)
        expected_rows = self.n + 1
        expected_shape = (expected_rows, 3)
        
        if val_arr.shape != expected_shape:
            print(f"Auto-fixing polynomial coefficients: Expected {expected_shape}, got {val_arr.shape}")
            
            new_coeffs = np.zeros(expected_shape)
            
            rows_to_copy = min(val_arr.shape[0], expected_rows)
            cols_to_copy = min(val_arr.shape[1], 3)
            
            if val_arr.ndim == 1:
                 limit = min(len(val_arr), expected_rows * 3)
                 new_coeffs.flat[:limit] = val_arr[:limit]
            else:
                 new_coeffs[:rows_to_copy, :cols_to_copy] = val_arr[:rows_to_copy, :cols_to_copy]
            
            self._coefficients = new_coeffs
        else:
            self._coefficients = val_arr
    
    def term_matrix(self):
        matrix = []
        for power in range(self.n + 1):
            if power == 0:
                row = [
                    lambda x, y, z: 1.0,
                    lambda x, y, z: 1.0,
                    lambda x, y, z: 1.0
                ]
            else:
                row = [
                    lambda x, y, z, p=power: np.clip(x ** p, -1e10, 1e10) if isinstance(x, np.ndarray) else min(max(x ** p, -1e10), 1e10),
                    lambda x, y, z, p=power: np.clip(y ** p, -1e10, 1e10) if isinstance(y, np.ndarray) else min(max(y ** p, -1e10), 1e10),
                    lambda x, y, z, p=power: np.clip(z ** p, -1e10, 1e10) if isinstance(z, np.ndarray) else min(max(z ** p, -1e10), 1e10)
                ]
            matrix.append(row)
        
        return np.array(matrix, dtype=object)
    
    def random_coefficients(self):
        return np.random.randn(self.n + 1, 3)
        
    def create_exponential_prob_vector(self, n, decay):
        probs = np.exp(-decay * np.arange(n))
        probs = probs / probs.sum()
        return probs

    def generate(self, num_terms, safe_mode=True, max_power=2, max_coeff=0.5, clip_output=True):
        polynom_terms = []
        index_list = []
        coefficients = []
        
        for i in range(num_terms):
            if np.random.rand() < 0.8 or len(polynom_terms) == 0:
                idx = np.random.choice([0, 1, 2])
                
                if safe_mode:
                    safe_powers = [p for p in range(min(max_power + 1, len(self.terms)))]
                    safe_probs = self.term_power_probability[:len(safe_powers)]
                    safe_probs = safe_probs / safe_probs.sum()
                    jdx = np.random.choice(safe_powers, p=safe_probs)
                else:
                    jdx = np.random.choice(len(self.terms), p=self.term_power_probability)
                
                term = self.terms[jdx][idx]
                coeff = self.coefficients[jdx][idx]
                
                if safe_mode:
                    coeff = np.clip(coeff, -max_coeff, max_coeff)
                
                coefficients.append(coeff)
                index_list.append([idx, jdx])
                polynom_terms.append(lambda x, y, z, t=term, c=coeff: c * t(x, y, z))
        
        def final_polynom(x, y, z):
            if isinstance(x, np.ndarray):
                result = np.zeros_like(x, dtype=float)
            else:
                result = 0.0
            
            for term_func in polynom_terms:
                try:
                    with np.errstate(over='raise', invalid='raise'):
                        term_result = term_func(x, y, z)
                except (FloatingPointError, RuntimeWarning):
                    continue
                
                term_result = np.nan_to_num(term_result, nan=0.0,
                                           posinf=100.0, neginf=-100.0)
                result += term_result
            
            result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
            
            if clip_output and safe_mode:
                result = np.clip(result, -100.0, 100.0)
            
            return result
        
        return final_polynom, index_list, coefficients
    
    
    def reconstruct(self, index_list, coefficients, clamp_value=50.0, noise_strength=0.05):
        polynom_terms = []
       
        for (idx, jdx), coeff in zip(index_list, coefficients):
            term = self.terms[jdx][idx]
            polynom_terms.append(lambda x, y, z, t=term, c=coeff: c * t(x, y, z))
       
        def final_polynom(x, y, z):
            if isinstance(x, np.ndarray):
                result = np.zeros_like(x, dtype=float)
            else:
                result = 0.0
           
            for term_func in polynom_terms:
                with np.errstate(over='ignore', invalid='ignore'):
                    term_result = term_func(x, y, z)
                
                if isinstance(term_result, np.ndarray):
                    term_result = np.nan_to_num(term_result, nan=0.0,
                                               posinf=clamp_value, neginf=-clamp_value)
                elif not np.isfinite(term_result):
                    term_result = 0.0
                
                result += term_result
            
            if isinstance(result, np.ndarray):
                too_high = result > clamp_value
                too_low = result < -clamp_value
                
                if np.any(too_high):
                    noise = np.random.uniform(0, noise_strength * clamp_value, size=np.sum(too_high))
                    result[too_high] = clamp_value - noise
                    
                if np.any(too_low):
                    noise = np.random.uniform(0, noise_strength * clamp_value, size=np.sum(too_low))
                    result[too_low] = -clamp_value + noise
                
                result = np.nan_to_num(result, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
            else:
                if result > clamp_value:
                    result = clamp_value - np.random.uniform(0, noise_strength * clamp_value)
                elif result < -clamp_value:
                    result = -clamp_value + np.random.uniform(0, noise_strength * clamp_value)
            
            return result
       
        return final_polynom

    def reconstruct_from_list(self, indices_list, coefficients_list):
        polynoms = []
    
        for index_list, coefficient_list in zip(indices_list, coefficients_list):
            poly = self.reconstruct(index_list, coefficient_list)
            polynoms.append(poly)
    
        return polynoms
    
    def generate_multiple(self, n_polynoms, num_terms, safe_mode=True, max_power=2, max_coeff=0.5):
        polynoms = []
        indices = []
        coeffs = []
    
        for _ in range(n_polynoms):
            poly, index_list, coefficient_list = self.generate(
                num_terms,
                safe_mode=safe_mode,
                max_power=max_power,
                max_coeff=max_coeff
            )
            polynoms.append(poly)
            indices.append(index_list)
            coeffs.append(coefficient_list)
    
        return polynoms, indices, coeffs
    
    def decode_polynom(self, encoded_polynom):

        index_list = encoded_polynom['indices']
        coefficient_list = encoded_polynom['coefficients']
        
        return self.reconstruct(index_list, coefficient_list)
    
    def encode_multiple(self, indices_list, coefficients_list):
        encoded_polynoms = []
        for idx_list, coeff_list in zip(indices_list, coefficients_list):
            encoded = self.encode_polynom(idx_list, coeff_list)
            encoded_polynoms.append(encoded)
        return encoded_polynoms
    
    def decode_multiple(self, encoded_polynoms):
        polynoms = []
        for encoded in encoded_polynoms:
            poly = self.decode_polynom(encoded)
            polynoms.append(poly)
        return polynoms

    def derivative(self, encoded_polynom, variable='x'):
        index_list = encoded_polynom['indices']
        coefficient_list = encoded_polynom['coefficients']
        
        var_map = {'x': 0, 'y': 1, 'z': 2}
        target_idx = var_map[variable]
        
        deriv_terms = []
        
        for (idx, jdx), coeff in zip(index_list, coefficient_list):
            if idx != target_idx:
                continue
            power = jdx
            
            if power == 0:
                continue
            
            new_coeff = coeff * power
            new_power = power - 1
            
            deriv_terms.append({
                'idx': idx,
                'jdx': new_power,
                'coeff': new_coeff
            })
        
        def derivative_polynom(x, y, z):
            if isinstance(x, np.ndarray):
                result = np.zeros_like(x, dtype=float)
            else:
                result = 0.0
            
            for term_info in deriv_terms:
                idx = term_info['idx']
                jdx = term_info['jdx']
                coeff = term_info['coeff']
                
                term = self.terms[jdx][idx]
                
                result += coeff * term(x, y, z)
            
            return result
        
        return derivative_polynom

    def gradient(self, encoded_polynom):

        df_dx = self.derivative(encoded_polynom, variable='x')
        df_dy = self.derivative(encoded_polynom, variable='y')
        df_dz = self.derivative(encoded_polynom, variable='z')
        
        return df_dx, df_dy, df_dz
    
    
    def gradient_vector(self, encoded_polynom, x, y, z):

        df_dx, df_dy, df_dz = self.gradient(encoded_polynom)
        
        grad = np.array([
            df_dx(x, y, z),
            df_dy(x, y, z),
            df_dz(x, y, z)
        ])
        
        return grad
    def derivative_encoded(self, encoded_polynom, variable='x'):

        index_list = encoded_polynom['indices']
        coefficient_list = encoded_polynom['coefficients']
        
        var_map = {'x': 0, 'y': 1, 'z': 2}
        target_idx = var_map[variable]
        
        new_indices = []
        new_coeffs = []
        
        for (idx, jdx), coeff in zip(index_list, coefficient_list):
            if idx != target_idx or jdx == 0:
                continue
            
            power = jdx
            new_coeff = coeff * power
            new_power = power - 1
            
            new_indices.append([idx, new_power])
            new_coeffs.append(new_coeff)
        
        return {
            'indices': new_indices,
            'coefficients': new_coeffs,
            'n': self.n,
            'decay': self.decay
        }
    
    
    def second_derivative(self, encoded_polynom, var1='x', var2='x'):

        first_deriv_encoded = self.derivative_encoded(encoded_polynom, variable=var1)
        
        second_deriv = self.derivative(first_deriv_encoded, variable=var2)
        
        return second_deriv
    


def circle3d(
    n=10,
    r=1.0,
    theta=0,
    phi=0,
    m=(0,0,0),
    name="Circle",
    plot=False
    ):

    
    X_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x_coords = SCALING_FACTOR * r * np.cos(X_angles)
    y_coords = SCALING_FACTOR * r * np.sin(X_angles)
    z_coords = np.zeros(n)

    theta = np.deg2rad(theta)
    phi   = np.deg2rad(phi)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta),  np.cos(theta)]])
    Ry = np.array([[ np.cos(phi), 0, np.sin(phi)],
                   [0,           1, 0         ],
                   [-np.sin(phi),0, np.cos(phi)]])
    rotM = Ry @ Rx

    pts = np.vstack((x_coords, y_coords, z_coords)).T
    pts = pts @ rotM
    pts = pts + np.array(m)

    if plot:
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(pts[:,0], pts[:,1], pts[:,2], lw=2)
        max_val = np.max(np.abs(pts))
        ax.set_xlim(-max_val*1.2, max_val*1.2)
        ax.set_ylim(-max_val*1.2, max_val*1.2)
        ax.set_zlim(-max_val*1.2, max_val*1.2)
        ax.set_title(name)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.show()

    return pts


def create_cone(
    m=np.zeros(3, dtype=np.float32),
    n=1500,
    inner_radius=0.1,
    outer_radius=0.2,
    height=0.6,
    rot_theta=0.0,
    rot_phi=0.0,
    plot=False,
    name="Cone"
    ):
    
    

    inner_r = inner_radius * SCALING_FACTOR
    outer_r = outer_radius * SCALING_FACTOR
    h = height * SCALING_FACTOR
    
    ang = np.random.uniform(0, 2*np.pi, n)
    z = np.random.uniform(m[2], m[2] + h, n)
    
    r = inner_r + (outer_r - inner_r) * ((z - m[2]) / h)
    
    x = m[0] + r * np.cos(ang)
    y = m[1] + r * np.sin(ang)
    
    th = np.deg2rad(rot_theta)
    ph = np.deg2rad(rot_phi)
    
    Rx = np.array([[1,         0,          0       ],
                   [0, np.cos(th), -np.sin(th)],
                   [0, np.sin(th),  np.cos(th)]])
    Ry = np.array([[ np.cos(ph), 0, np.sin(ph)],
                   [          0, 1,          0],
                   [-np.sin(ph), 0, np.cos(ph)]])
    rotM = Ry @ Rx
    
    points = np.column_stack((x, y, z))
    points = points @ rotM.T
    
    if plot:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2], s=2)
        ax.set(title=name)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        plt.show()
    
    return points


def blob_positions(
    n = 10,
    m = np.zeros(3),
    r = 1.0,
    scaling_factor = 1.0,
    plot=False,
    name="Blob"
    ):


    pos_norm = np.random.normal(size=(n, 3))
    pos_norm /= np.linalg.norm(pos_norm, axis=1, keepdims=True)
    
    u = np.random.rand(n, 1)
    r_scaling = r * u**(1/3)
    
    pts = m + pos_norm * r_scaling * scaling_factor

    if plot:
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')

        ax.scatter(pts[:,0], pts[:,1], pts[:,2])

        lim = 1.1*np.abs(pts).max()
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim),
               title=name, xlabel='X', ylabel='Y', zlabel='Z')
        plt.show()
    return pts


def create_Grid(m=np.zeros(3, dtype=np.float32),
                grid_size_list=[28, 15, 10],
                rot_theta=0.0,
                rot_phi=0.0,
                plot=False,
               name="Grid"):


    
    th = np.deg2rad(rot_theta)
    ph = np.deg2rad(rot_phi)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(th), -np.sin(th)],
                   [0, np.sin(th),  np.cos(th)]])
    Ry = np.array([[ np.cos(ph), 0, np.sin(ph)],
                   [          0, 1,          0],
                   [-np.sin(ph), 0, np.cos(ph)]])
    rotM = Ry @ Rx

    node_layers = []
    for d, k in enumerate(grid_size_list):
        pts = np.array([[d + m[0], w + m[1], n + m[2]]
                        for w in range(k) for n in range(k)])
        node_layers.append(pts @ rotM.T)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for layer_pts in node_layers:
            ax.scatter(layer_pts[:, 0], layer_pts[:, 1], layer_pts[:, 2], s=1)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set(title=name)
        try:
            ax.set_box_aspect((1,1,1))
        except AttributeError:
            set_axes_equal(ax)
        plt.show()

    return node_layers


def wave_collapse_old(mask=None,
                  dims=(2,2,2),
                  sparse_holes = 0,
                  type_array=[0,1],
                  probability_vector=np.array([0.3,0.7]),
                  start_pos=np.zeros(3),
                  sparsity_factor=0.7,
                seed=42):
    
    
    
    
    random.seed(seed+3)
    np.random.seed(seed+7)

    probability_vector = np.asarray(probability_vector, dtype=float)
    probability_vector /= probability_vector.sum()
    
    if mask is None:
        mask = np.zeros(dims, dtype=np.int32)
    dims=mask.shape
    
    entropy_matrix = np.full(shape=dims, fill_value=1000.0, dtype=np.float32)
    
    collapsed_nodes = np.full(shape=dims, fill_value=-1, dtype=np.int32)
    
    
    D0, D1, D2 = mask.shape
    
    
    if(sparse_holes!=0):
        for _ in range(0,sparse_holes,1):
            x = random.randint(0,D0-1)
            y = random.randint(0,D1-1)
            z = random.randint(0,D2-1)
            mask[x][y][z] = -1
        
    def check_neighbors(i, j, k):
        undef_neighbors = []
        def_neighbors = []
        shifts = [(dx, dy, dz)
              for dx in (-1, 0, 1)
              for dy in (-1, 0, 1)
              for dz in (-1, 0, 1)
              if not (dx == dy == dz == 0)]
        for di, dj, dk in shifts:
            ni, nj, nk = i+di, j+dj, k+dk
            if 0 <= ni < D0 and 0 <= nj < D1 and 0 <= nk < D2:
                if mask[ni, nj, nk] == 0:
                    undef_neighbors.append((ni, nj, nk))
                elif mask[ni, nj, nk] == 1:
                    def_neighbors.append((ni, nj, nk))
                else:
                    continue

                    
        return undef_neighbors, def_neighbors
    

    def entropy(p, base=None):
        p = np.asarray(p, dtype=float)
        p += 0.0001
        p/=np.sum(p)
        log_p = np.log(p) if base is None else np.log(p) / np.log(base)
        return -np.sum(p * log_p)
    
    def collapse(i, j, k):
        if(mask[i][j][k]):
            pass
        

        
        l = len(probability_vector)
        l1 = int((1-sparsity_factor)*l)

        list_of_uncollapsed_neighbors, list_of_collapsed_neighbors = check_neighbors(i,j,k)
        
        neighbor_labels = []
        for x,y,z in list_of_collapsed_neighbors:
            neighbor_labels.append(collapsed_nodes[x][y][z])
        
        
        private_probability_vector = probability_vector.copy()

        original_pv = private_probability_vector.copy()
        
        for m in neighbor_labels:
            p = private_probability_vector[m] / (l + l1)
            p_self = l1 * p
            p_rest = p
            p_redistribution = np.full(shape=(l,),fill_value=p_rest)
            p_redistribution[m] += p_self
            private_probability_vector[m] = 0.0
            private_probability_vector+=p_redistribution
        private_probability_vector/=private_probability_vector.sum()
        private_probability_vector=(private_probability_vector+2*original_pv)/3
        choice = np.random.choice(type_array,p=private_probability_vector)
        collapsed_nodes[i][j][k] = choice
        mask[i][j][k]=1
        entropy_matrix[i][j][k] = 0.0


        for ni, nj, nk in list_of_uncollapsed_neighbors:
            _, coll_nbrs = check_neighbors(ni, nj, nk)
            neigh_labels = [collapsed_nodes[x,y,z] for x,y,z in coll_nbrs]

            allowed = np.setdiff1d(type_array, neigh_labels)

            p = np.zeros_like(probability_vector, dtype=float)
            idxs = [type_array.index(a) for a in allowed]
            p[idxs] = probability_vector[idxs]
            if p.sum() > 0:
                p /= p.sum()
            else:
                p[:] = 1 / len(p)
            entropy_matrix[ni, nj, nk] = entropy(p, base=2)
            
            
    def _next():
        candidates = np.argwhere(mask == 0)
        if candidates.size == 0:
            return None
        valid = [(tuple(idx), entropy_matrix[tuple(idx)])
                 for idx in candidates
                 if entropy_matrix[tuple(idx)] > 0]
        if not valid:
            return None
        next_idx = min(valid, key=lambda x: x[1])[0]
        return next_idx
    
    while True:
        nxt = _next()
        if nxt is None:
            break
        collapse(*nxt)

    return np.array(collapsed_nodes)


def wave_collapse(dims, type_array, probability_vector,
                       sparsity_factor=.7, seed=0, sparse_holes=0):
    NEIGHBOR_OFFSETS = np.array([(dx,dy,dz)
      for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)
      if (dx,dy,dz)!=(0,0,0)], dtype=np.int8)
    random.seed(seed+3); np.random.seed(seed+7)
    probability_vector = np.asarray(probability_vector, float); probability_vector /= probability_vector.sum()
    mask  = np.zeros(dims, np.int8)
    if sparse_holes:
        idx = np.random.choice(mask.size, sparse_holes, replace=False)
        mask.ravel()[idx] = -1

    collapsed = -np.ones(dims, np.int16)
    H = np.full(dims, 999., np.float32)

    heap = [(H[i,j,k],i,j,k) for i in range(dims[0])
                               for j in range(dims[1])
                               for k in range(dims[2]) if mask[i,j,k]==0]
    heapq.heapify(heap)

    while heap:
        h,i,j,k = heapq.heappop(heap)
        if mask[i,j,k]:
            continue

        neigh = NEIGHBOR_OFFSETS + (i,j,k)
        good  = ((neigh[:,0] >= 0) & (neigh[:,0] < dims[0]) &
                 (neigh[:,1] >= 0) & (neigh[:,1] < dims[1]) &
                 (neigh[:,2] >= 0) & (neigh[:,2] < dims[2]))
        neigh = neigh[good]
        labels = collapsed[tuple(neigh.T)]
        labels = labels[labels >= 0]

        if labels.size:
            counts = np.bincount(labels, minlength=len(type_array))
            boost  = counts * (1-sparsity_factor)/len(type_array)
            p = probability_vector + boost
            p /= p.sum()
        else:
            p = probability_vector

        choice = np.random.choice(type_array, p=p)
        collapsed[i,j,k] = choice
        mask[i,j,k] = 1
        H[i,j,k] = 0.0

        for ni,nj,nk in neigh:
            if mask[ni,nj,nk]==0:
                H[ni,nj,nk] = 1.0
                heapq.heappush(heap,(H[ni,nj,nk],ni,nj,nk))

    return collapsed


def generate_cube(grid_size_list=(10, 10, 10)):
    nx, ny, nz = grid_size_list
    xs = np.linspace(-1, 1, nx)
    ys = np.linspace(-1, 1, ny)
    zs = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    return np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T


def transform_points(
    points,
    m = np.zeros(3, dtype=float),
    rot_theta = 0.0,
    rot_phi = 0.0,
    transform_matrix = np.eye(3),
    plot = False
    ):
    pts = np.asarray(points, dtype=float)
    
    th = np.deg2rad(rot_theta)
    ph = np.deg2rad(rot_phi)
    Rx = np.array([[1,          0,           0],
                   [0, np.cos(th), -np.sin(th)],
                   [0, np.sin(th),  np.cos(th)]])
    Ry = np.array([[ np.cos(ph), 0, np.sin(ph)],
                   [          0, 1,          0],
                   [-np.sin(ph), 0, np.cos(ph)]])
    rotM = Ry @ Rx
    
    rotated = (rotM @ pts.T).T
            
    transformed = (transform_matrix @ rotated.T).T
    
    result = transformed + np.asarray(m, dtype=float)
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(result[:,0], result[:,1], result[:,2], s=2)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        plt.show()
    
    return result


def field(
        positions: np.ndarray,
        f1 = lambda x, y, z: np.zeros_like(x),
        f2 = lambda x, y, z: np.zeros_like(x),
        f3 = lambda x, y, z: np.zeros_like(x),
        plot=False,
        normalize = False,
        color_by_length = False,
        cmap = 'viridis',
        arrow_length = 0.1
    ):
    
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    v1 = f1(x, y, z)
    v2 = f2(x, y, z)
    v3 = f3(x, y, z)
    vectors = np.column_stack((v1, v2, v3))
    if plot:
        lengths = np.linalg.norm(vectors, axis=1)
        U, V, W = v1.copy(), v2.copy(), v3.copy()
        if normalize:
            nonzero = lengths > 0
            U[nonzero] /= lengths[nonzero]
            V[nonzero] /= lengths[nonzero]
            W[nonzero] /= lengths[nonzero]
            U *= arrow_length
            V *= arrow_length
            W *= arrow_length
        else:
            U *= arrow_length
            V *= arrow_length
            W *= arrow_length

        color_args = {}
        if color_by_length:
            normed = (lengths - lengths.min()) / (lengths.ptp() if lengths.ptp()>0 else 1)
            cmap_obj = plt.get_cmap(cmap)
            colors = cmap_obj(normed)
            color_args['color'] = colors

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(
            x, y, z,
            U, V, W,
            **color_args,
            length=1.0,
            normalize=False
        )
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        plt.show()
    return vectors


def generate_torus(
        R = 1.0,
        r = 0.3,
        grid_size_list = (100, 40),
    rot_theta=0,
    rot_phi=0,
    plot=False
    ):
    n_theta, n_phi = grid_size_list

    
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phi   = np.linspace(0.0, 2.0 * np.pi, n_phi,   endpoint=False)

    a, b = np.meshgrid(theta, phi, indexing='ij')

    x = (R + r * np.cos(b)) * np.cos(a)
    y = (R + r * np.cos(b)) * np.sin(a)
    z =  r * np.sin(b)

    return transform_points(np.vstack((x.ravel(), y.ravel(), z.ravel())).T,
                            rot_theta=rot_theta, rot_phi=rot_phi, plot=plot)


def simulate_vector_field_flow(
            initial_positions = np.zeros((3,3)),
            dt = 0.001,
            f1 = lambda x, y, z: np.zeros_like(x),
            f2 = lambda x, y, z: np.zeros_like(x),
            f3 = lambda x, y, z: np.zeros_like(x),
            num_steps = 1,
            plot= True,
            scatter_size = 1.0,
            create_object = False
            ):
    
    positions = initial_positions.copy()
    snapshots = []
    
    if(create_object):
        snapshots.append(positions)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    
    for _ in range(num_steps):
        if plot:
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=scatter_size)
        velocity = field(positions, f1=f1, f2=f2, f3=f3, plot=False)
        positions = positions + dt * velocity
        if(create_object):
            snapshots.append(positions)

    if plot:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    if(create_object):
        return snapshots
    return positions

def field_flow_iteration(
                        positions=np.zeros(3),
                        f1 = lambda x, y, z: np.zeros_like(x),
                        f2 = lambda x, y, z: np.zeros_like(x),
                        f3 = lambda x, y, z: np.zeros_like(x)):
    velocity = field(
        positions=positions,
        plot=False,
        f1=f1,
        f2=f2,
        f3=f3
    )
    return positions + dt * velocity


def add_local_attractor(
    m = np.zeros(3),
    k = 15.0,
    sigma = 0.9,
    f1 = lambda x, y, z: np.zeros_like(x),
    f2 = lambda x, y, z: np.zeros_like(x),
    f3 = lambda x, y, z: np.zeros_like(x),
    repulsive=False
):
    x0, y0, z0 = m
    s = 1.0 if repulsive else -1.0

    def w(r):
        return s*np.exp(-0.5*(r/sigma)**2)

    def f1_mod(x, y, z):
        dx = x - x0
        r = np.sqrt(dx*dx + (y-y0)**2 + (z-z0)**2)
        sink = -k * s * w(r) * dx
        return f1(x, y, z) + sink

    def f2_mod(x, y, z):
        dy = y - y0
        r = np.sqrt((x-x0)**2 + dy*dy + (z-z0)**2)
        sink = -k * s * w(r) * dy
        return f2(x, y, z) + sink

    def f3_mod(x, y, z):
        dz = z - z0
        r = np.sqrt((x-x0)**2 + (y-y0)**2 + dz*dz)
        sink = -k * s * w(r) * dz
        return f3(x, y, z) + sink

    return f1_mod, f2_mod, f3_mod

def find_positions_by_type(
    volume,
    types
):
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D array, got ndim={volume.ndim}")

    positions_by_type: Dict[int, np.ndarray] = {}
    for t in types:
        coords = np.argwhere(volume == t)
        positions_by_type[t] = coords

    return positions_by_type
def anisotropic_wave_collapse(
    dims,
    type_array,
    probability_vector,
    sparsity_factor=0.7,
    seed=0,
    sparse_holes=0,
    vertical_bias=0.9,
    layer_boundaries=None,
    layer_type_modifiers=None
):
    
    
    NEIGHBOR_OFFSETS = []
    NEIGHBOR_WEIGHTS = []
    
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            NEIGHBOR_OFFSETS.append((dx, dy, 0))
            weight = sparsity_factor if (dx == 0 or dy == 0) else sparsity_factor * 0.7
            NEIGHBOR_WEIGHTS.append(weight)
    
    vertical_sparsity = min(1.0, sparsity_factor + vertical_bias)
    NEIGHBOR_OFFSETS.append((0, 0, 1))
    NEIGHBOR_WEIGHTS.append(vertical_sparsity)
    NEIGHBOR_OFFSETS.append((0, 0, -1))
    NEIGHBOR_WEIGHTS.append(vertical_sparsity)
    
    NEIGHBOR_OFFSETS = np.array(NEIGHBOR_OFFSETS, dtype=np.int8)
    NEIGHBOR_WEIGHTS = np.array(NEIGHBOR_WEIGHTS, dtype=np.float32)
    
    random.seed(seed + 3)
    np.random.seed(seed + 7)
    probability_vector = np.asarray(probability_vector, float)
    probability_vector /= probability_vector.sum()
    
    mask = np.zeros(dims, np.int8)
    if sparse_holes:
        idx = np.random.choice(mask.size, sparse_holes, replace=False)
        mask.ravel()[idx] = -1
    
    collapsed = -np.ones(dims, np.int16)
    H = np.full(dims, 999., np.float32)
    
    heap = [(H[i,j,k], i, j, k) for i in range(dims[0])
                                  for j in range(dims[1])
                                  for k in range(dims[2]) if mask[i,j,k] == 0]
    heapq.heapify(heap)
    
    def get_layer_idx(z):
        if layer_boundaries is None:
            return None
        for idx, boundary in enumerate(layer_boundaries):
            if z < boundary:
                return idx
        return len(layer_boundaries)
    
    def get_probs_for_layer(z):
        layer_idx = get_layer_idx(z)
        if layer_idx is not None and layer_type_modifiers and layer_idx in layer_type_modifiers:
            p = np.array(layer_type_modifiers[layer_idx], dtype=float)
            p /= p.sum()
            return p
        return probability_vector.copy()
    
    while heap:
        h, i, j, k = heapq.heappop(heap)
        if mask[i, j, k]:
            continue
        
        neighbor_labels = []
        neighbor_weights_used = []
        
        for offset, weight in zip(NEIGHBOR_OFFSETS, NEIGHBOR_WEIGHTS):
            ni, nj, nk = i + offset[0], j + offset[1], k + offset[2]
            
            if not (0 <= ni < dims[0] and 0 <= nj < dims[1] and 0 <= nk < dims[2]):
                continue
            
            if mask[ni, nj, nk] == 1:
                label = collapsed[ni, nj, nk]
                neighbor_labels.append(label)
                neighbor_weights_used.append(weight)
        
        p = get_probs_for_layer(k)
        
        if neighbor_labels:
            counts = np.zeros(len(type_array), dtype=float)
            for label, weight in zip(neighbor_labels, neighbor_weights_used):
                if label in type_array:
                    idx = type_array.index(label)
                    counts[idx] += weight
            
            if counts.sum() > 0:
                counts /= counts.sum()
            
            l = len(type_array)
            l1 = int((1 - sparsity_factor) * l)
            
            boost = counts * (1 - sparsity_factor) / (l + l1)
            boost_self = boost * l1
            boost_rest = boost
            
            redistribution = np.full(l, boost_rest.mean())
            for idx in range(l):
                redistribution[idx] += boost_self[idx]
            
            p = p + redistribution
            p = np.maximum(p, 0)
            if p.sum() > 0:
                p /= p.sum()
            else:
                p = get_probs_for_layer(k)
        
        choice = np.random.choice(type_array, p=p)
        collapsed[i, j, k] = choice
        mask[i, j, k] = 1
        H[i, j, k] = 0.0
        
        for offset in NEIGHBOR_OFFSETS:
            ni, nj, nk = i + offset[0], j + offset[1], k + offset[2]
            if (0 <= ni < dims[0] and 0 <= nj < dims[1] and 0 <= nk < dims[2]
                and mask[ni, nj, nk] == 0):
                H[ni, nj, nk] = 1.0
                heapq.heappush(heap, (H[ni, nj, nk], ni, nj, nk))
    
    return collapsed

def wave_collapse_bio(
    dims,
    type_array,
    probability_vector,
    sparsity_factor=0.7,
    seed=0,
    sparse_holes=0
):

    return anisotropic_wave_collapse(
        dims=dims,
        type_array=type_array,
        probability_vector=probability_vector,
        sparsity_factor=sparsity_factor,
        seed=seed,
        sparse_holes=sparse_holes,
        vertical_bias=0.25,
        layer_boundaries=None,
        layer_type_modifiers=None
    )

def cluster_and_flow(
    grid_size,
    m,
    rot_theta,
    rot_phi,
    transform_matrix,
    wave_params,
    types,
    flow_functions,
    dt,
    num_steps,
    old=True,
    biological=False,
    plot_clusters=False,
    title="Cluster-Flows",
    vertical_bias=0.25,
    layer_boundaries=None,
    layer_type_modifiers=None
):
    grid = generate_cube(grid_size)
    pts = transform_points(
        grid, m=m, rot_theta=rot_theta,
        rot_phi=rot_phi, transform_matrix=transform_matrix,
        plot=False
    )
    
    if biological:
        neuron_type_array = anisotropic_wave_collapse(
            dims=grid_size,
            type_array=types,
            probability_vector=wave_params['probability_vector'],
            sparsity_factor=wave_params['sparsity_factor'],
            seed=wave_params.get('seed', 0),
            sparse_holes=wave_params['sparse_holes'],
            vertical_bias=vertical_bias,
            layer_boundaries=layer_boundaries,
            layer_type_modifiers=layer_type_modifiers
        )
    elif old:
        neuron_type_array = wave_collapse_old(
            dims=grid_size,
            sparse_holes=wave_params['sparse_holes'],
            sparsity_factor=wave_params['sparsity_factor'],
            probability_vector=wave_params['probability_vector'],
            type_array=types
        )
    else:
        neuron_type_array = wave_collapse(
            dims=grid_size,
            sparse_holes=wave_params['sparse_holes'],
            sparsity_factor=wave_params['sparsity_factor'],
            probability_vector=wave_params['probability_vector'],
            type_array=types
        )
    
    idx = find_positions_by_type(neuron_type_array, types)
    
    nx, ny, nz = grid_size
    slice_size = ny * nz
    row_size = nz
    arrays = []
    
    for t in types:
        tripels = idx[t]
        flat_idx = tripels[:, 0] * slice_size + tripels[:, 1] * row_size + tripels[:, 2]
        arrays.append(pts[flat_idx])
    
    clusters = []
    for (f1, f2, f3), cluster_pts in zip(flow_functions, arrays):
        final_positions = simulate_vector_field_flow(
            initial_positions=cluster_pts,
            dt=dt, f1=f1, f2=f2, f3=f3,
            num_steps=num_steps,
            plot=False, scatter_size=1.0,
            create_object=False
        )
        clusters.append(final_positions)
    
    if plot_clusters:
        plot_point_clusters(clusters, title=title)
    
    return clusters

def cluster_and_flow_flexible(
    points,
    grid_size,
    wave_params,
    types,
    flow_functions,
    dt,
    num_steps,
    old=True,
    biological=False,
    plot_clusters=False,
    title="Cluster-Flows",
    vertical_bias=0.25,
    layer_boundaries=None,
    layer_type_modifiers=None
):
    

    if biological:
        neuron_type_array = anisotropic_wave_collapse(
            dims=grid_size,
            type_array=types,
            probability_vector=wave_params['probability_vector'],
            sparsity_factor=wave_params['sparsity_factor'],
            seed=wave_params.get('seed', 0),
            sparse_holes=wave_params['sparse_holes'],
            vertical_bias=vertical_bias,
            layer_boundaries=layer_boundaries,
            layer_type_modifiers=layer_type_modifiers
        )
    else:
        neuron_type_array = wave_collapse(
            dims=grid_size,
            type_array=types,
            probability_vector=wave_params['probability_vector'],
            sparsity_factor=wave_params['sparsity_factor'],
            seed=wave_params.get('seed', 0),
            sparse_holes=wave_params['sparse_holes']
        )
    
    points = np.asarray(points)
    n_points = len(points)
    
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins + 1e-10
    
    normalized = (points - mins) / ranges
    
    nx, ny, nz = grid_size
    grid_indices = np.floor(normalized * [nx-1, ny-1, nz-1]).astype(int)
    grid_indices = np.clip(grid_indices, [0,0,0], [nx-1, ny-1, nz-1])
    
    point_types = neuron_type_array[grid_indices[:, 0],
                                    grid_indices[:, 1],
                                    grid_indices[:, 2]]
    
    clusters = []
    for t in types:
        mask = (point_types == t)
        cluster_pts = points[mask]
        clusters.append(cluster_pts)
    
    final_clusters = []
    for (f1, f2, f3), cluster_pts in zip(flow_functions, clusters):
        if len(cluster_pts) == 0:
            final_clusters.append(cluster_pts)
            continue
            
        final_positions = simulate_vector_field_flow(
            initial_positions=cluster_pts,
            dt=dt, f1=f1, f2=f2, f3=f3,
            num_steps=num_steps,
            plot=False
        )
        final_clusters.append(final_positions)
    
    if plot_clusters:
        plot_point_clusters(final_clusters, title=title)
    
    return final_clusters


def compute_rotational_similarity_weights(positions, scale_ex=1.0, scale_in=1.0):
    pos = np.array(positions)
    x = pos[:, 0]
    y = pos[:, 1]
    
    vectors = np.column_stack((-y, x, np.zeros_like(x)))
    
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors_norm = vectors / norms
    
    similarity_matrix = vectors_norm @ vectors_norm.T
    
    weights = np.zeros_like(similarity_matrix)
    
    mask_ex = similarity_matrix > 0
    weights[mask_ex] = similarity_matrix[mask_ex] * scale_ex
    
    mask_in = similarity_matrix < 0
    weights[mask_in] = similarity_matrix[mask_in] * scale_in
    
    return weights
def create_CCW(
    positions=None,
    model='iaf_psc_alpha',
    neuron_params=None,
    plot=False,
    syn_model='static_synapse',
    
    weight_ex=30.0,
    delay_ex=1.0,
    weight_in_factor=1.0,
    k=10.0,
    
    radius=5.0,
    n_neurons=100,
    center=np.zeros(3),
    rot_theta=0.0,
    rot_phi=0.0,
    
    bidirectional=False,
    use_index_based=False
    ):
    
    safe_params = neuron_params if neuron_params else {}
    
    t = np.linspace(0, 2*np.pi, n_neurons, endpoint=False)
    local_x = radius * np.cos(t)
    local_y = radius * np.sin(t)
    local_z = np.zeros(n_neurons)
    
    local_points = np.column_stack((local_x, local_y, local_z))
    
    world_points = apply_transform(local_points, rot_theta, rot_phi, center)
    
    nodes = nest.Create(model, positions=nest.spatial.free(pos=world_points.tolist()), params=safe_params)
    
    if plot:
        nest.PlotLayer(nodes)

    if use_index_based:
        m_neighbors = int(k)
        connect_neighbors_by_index(
            nodes,
            n_neighbors=m_neighbors,
            weight_ex=weight_ex,
            weight_in=weight_ex * weight_in_factor,
            delay=delay_ex
        )
        print(f"CCW (Index-Based): Connected {m_neighbors} prev/next neighbors.")
        
    else:
        
        weights_flat = compute_angular_similarity_weights(
            local_points,
            weight_ex=weight_ex,
            weight_in=weight_ex * weight_in_factor,
            bidirectional=bidirectional
        )
        
        delays_flat = np.ones_like(weights_flat) * float(delay_ex)
        
        gids = np.array(nodes.get("global_id"))
        N = len(gids)
        sources = np.repeat(gids, N)
        targets = np.tile(gids, N)
        
        mask = sources != targets
        
        nest.Connect(
            sources[mask],
            targets[mask],
            {'rule': 'one_to_one'},
            {
                'synapse_model': syn_model,
                'weight': weights_flat[mask],
                'delay': delays_flat[mask]
            }
        )
        mode = "Bidirectional" if bidirectional else "Unidirectional (Vector)"
        print(f"CCW ({mode}): Connected using similarity matrix.")

    return nodes


def CCW_spike_recorder(ccw):
        

    length = len(ccw)
    recorder_list = []
    for i, neuron in enumerate(ccw):
        theta = (i/length) * 2*np.pi
        spikerecorder = nest.Create("spike_recorder")
        nest.Connect(neuron, spikerecorder)
        recorder_list.append((theta,spikerecorder))
    return recorder_list

def connect_cone(
    cone_points=None,
    model='iaf_psc_alpha',
    neuron_params=None,
    
    k=10.0,
    weight_ex=30.0,
    
    n_neurons=100,
    radius_top=1.0,
    radius_bottom=5.0,
    height=10.0,
    center=np.zeros(3),
    rot_theta=0.0,
    rot_phi=0.0,
    
    bidirectional=False,
    use_index_based=False,
    
    **kwargs
    ):
    
    safe_params = neuron_params if neuron_params else {}
    
    
    z_local = np.linspace(0, height, n_neurons)
    r_at_z = radius_bottom + (radius_top - radius_bottom) * (z_local / height)
    
    phi_local = np.linspace(0, 4 * np.pi, n_neurons)
    
    local_x = r_at_z * np.cos(phi_local)
    local_y = r_at_z * np.sin(phi_local)
    
    local_points = np.column_stack((local_x, local_y, z_local))
    
    world_points = apply_transform(local_points, rot_theta, rot_phi, center)
    
    nodes = nest.Create(model, positions=nest.spatial.free(pos=world_points.tolist()), params=safe_params)
    
    w_ex = weight_ex if weight_ex else 10.0
    w_in = w_ex * 1.0
    
    if use_index_based:
        m_neighbors = int(k)
        connect_neighbors_by_index(
            nodes,
            n_neighbors=m_neighbors,
            weight_ex=w_ex,
            weight_in=-abs(w_ex),
            delay=1.0
        )
        print(f"Cone (Index-Based): Connected linear neighbors (Spiral).")
        
    else:
        
        weights_flat = compute_angular_similarity_weights(
            local_points,
            weight_ex=w_ex,
            weight_in=abs(w_ex * 2.0),
            bidirectional=bidirectional
        )
        
        delays_flat = np.ones_like(weights_flat) * 1.0
        
        gids = np.array(nodes.get("global_id"))
        N = len(gids)
        sources = np.repeat(gids, N)
        targets = np.tile(gids, N)
        mask = sources != targets
        
        nest.Connect(
            sources[mask], targets[mask],
            {'rule': 'one_to_one'},
            {'synapse_model': 'static_synapse', 'weight': weights_flat[mask], 'delay': delays_flat[mask]}
        )
        print(f"Cone (Similarity): Connected based on angular alignment.")

    return nodes


def connect_cone_ccw(
        cone,
        ccw,
        syn_spec_strong={'synapse_model':'static_synapse', 'weight':10.0, 'delay':1.0},
        syn_spec_weak  ={'synapse_model':'static_synapse', 'weight': 2.0, 'delay':1.0},
        angle_width_deg=15.0
    ):


    
    conn_spec   = {'rule': 'one_to_one'}
    angle_width = np.deg2rad(angle_width_deg)


    pos_ccw  = nest.GetPosition(ccw)
    pos_cone = nest.GetPosition(cone)

    theta_ccw  = [ (np.arctan2(y, x) + 2*np.pi) % (2*np.pi) for x, y, _ in pos_ccw ]
    theta_cone = [ (np.arctan2(y, x) + 2*np.pi) % (2*np.pi) for x, y, _ in pos_cone ]

    for idx_i, i in enumerate(ccw):
        ti = theta_ccw[idx_i]
        for idx_j, j in enumerate(cone):
            tj = theta_cone[idx_j]
            delta = abs(tj - ti)
            delta = min(delta, 2*np.pi - delta)
            if delta < angle_width:
                nest.Connect(j, i, conn_spec, syn_spec_strong)
            else:
                nest.Connect(j, i, conn_spec, syn_spec_weak)


def create_blob_population(
    positions,
    neuron_type="iaf_psc_alpha",
    neuron_params=None,
    conn_ex={'rule': 'pairwise_bernoulli', 'p': 0.8, 'allow_autapses': False},
    syn_ex={'synapse_model': 'static_synapse', 'weight': 2.0, 'delay': 1.0},
    conn_in={'rule': 'pairwise_bernoulli', 'p': 0.2, 'allow_autapses': False},
    syn_in={'synapse_model': 'static_synapse','weight': -10.0,'delay': 1.0},
    plot=False
    ):
    
    safe_params = neuron_params if neuron_params else {}
    
    blob_pop = nest.Create(
        neuron_type,
        positions.shape[0],
        positions=nest.spatial.free(positions.tolist()),
        params=safe_params
    )

    nest.Connect(blob_pop, blob_pop, conn_ex, syn_ex)
    nest.Connect(blob_pop, blob_pop, conn_in, syn_in)

    if plot:
        nest.PlotLayer(blob_pop)
    return blob_pop


def blob(
    n=100,
    m=np.zeros(3),
    r=1.0,
    neuron_params=None,
    scaling_factor=1.0,
    plot=False,
    neuron_type="iaf_psc_alpha"
    ):
    
    pos = blob_positions(n=n, m=m, r=1.0, scaling_factor=SCALING_FACTOR)
    
    blob_pop = create_blob_population(
        positions=pos,
        neuron_type=neuron_type,
        neuron_params=neuron_params,
        plot=plot
    )
    return blob_pop

def connect_blob_cone(
    blob,
    cone,
    m = np.zeros(3),
    angle_width_deg = 10.0,
    conn_ex_c = {'rule': 'pairwise_bernoulli', 'p': 0.6, 'allow_autapses': False},
    conn_in_c  = {'rule': 'pairwise_bernoulli', 'p': 0.4, 'allow_autapses': False},
    generic_conn = {'rule': 'pairwise_bernoulli', 'p': 0.01, 'allow_autapses': False},
    generic_syn = {'synapse_model': 'static_synapse', 'weight': 1.0, 'delay': 1.5}
):


    nest.Connect(blob, cone, generic_conn, generic_syn)


    half = np.deg2rad(angle_width_deg)
    pos_blob = np.array(nest.GetPosition(blob))
    pos_cone = np.array(nest.GetPosition(cone))
    th_blob  = (np.arctan2(pos_blob[:,1],pos_blob[:,0]) + 2*np.pi)% (2*np.pi)
    th_cone  = (np.arctan2(pos_cone[:,1],pos_cone[:,0]) + 2*np.pi)% (2*np.pi)
    blob_ids = nest.GetStatus(blob, 'global_id')
    cone_ids = nest.GetStatus(cone, 'global_id')

    pre_exc, post_exc, w_exc, d_exc = [], [], [], []
    pre_inh, post_inh, w_inh, d_inh = [], [], [], []

    for bid, tb, pB in zip(blob_ids, th_blob, pos_blob):
        for cid, tc, pC in zip(cone_ids, th_cone, pos_cone):
            delta = abs(tc - tb)
            delta = min(delta, 2*np.pi - delta)
            if delta <= half:
                pre_exc.append(cid)
                post_exc.append(bid)

                dist = np.linalg.norm(pC - pB)
                w_exc.append(2.4 + 1.2 * dist)
                d_exc.append(1.0)
            else:
                pre_inh.append(cid)
                post_inh.append(bid)
                dx, dy = pC[0] - m[0], pC[1] - m[1]
                w_inh.append(1.0 + dx*dx + dy*dy)
                raw = np.random.exponential(scale=2.0) * np.linalg.norm(pC - pB)
                d_inh.append(max(raw, 1.0))

    if pre_exc:
        nest.Connect(
            pre_exc, post_exc,
            {'rule':'one_to_one'},
            {'synapse_model':'static_synapse',
             'weight': np.array(w_exc),
             'delay':  np.array(d_exc)}
        )
    if pre_inh:
        nest.Connect(
            pre_inh, post_inh,
            {'rule':'one_to_one'},
            {'synapse_model':'static_synapse',
             'weight': np.array(w_inh),
             'delay':  np.array(d_inh)}
        )


def grid2visual(
                grid,
                K=10,
                m=np.array([0.0,0.0,0.0]),
                syn_dict_ex = {
                    "synapse_model": "static_synapse",
                    "weight": 50.0,
                    "delay": 1.0
                },
                syn_dict_in = {
                    "synapse_model": "static_synapse",
                    "weight": -40.0,
                    "delay": 1.0
                },
                conn_dict_ex = {
                    "rule": "pairwise_bernoulli",
                    "p": 0.7,
                    "allow_autapses": False
                },
                syn_spec_ex = {
                    "synapse_model": "static_synapse",
                    "weight": 30.0,
                    "delay": 1.5
                },
                conn_dict_inh = {
                    "rule": "pairwise_bernoulli",
                    "p": 0.3,
                    "allow_autapses": False
                },
                syn_spec_inh = {
                    "synapse_model": "static_synapse",
                    "weight": -60.0,
                    "delay": 1.5
                },
                conn_dict_layer = {
                    "rule": "all_to_all",
                    "allow_autapses": False
                },
                syn_dict_layer = {
                    "synapse_model": "stdp_synapse",
                    "alpha": 1.0,
                    "lambda": 0.01,
                    "tau_plus": 20.0
                }
    ):


    
    length = 0
    populations = []
    poisson_generators = []
    noise_generators = []
    
    for i,layer in enumerate(grid):
        length+=1
        pop = nest.Create("iaf_psc_alpha",len(layer.tolist()),positions=nest.spatial.free(layer.tolist()))
        if i==0:
            for neuron in pop:
                ex = nest.Create("poisson_generator")
                noise = nest.Create("poisson_generator")
                nest.Connect(ex, neuron, syn_spec=syn_dict_ex)
                nest.Connect(noise, neuron, syn_spec=syn_dict_in)
                poisson_generators.append(ex)
                noise_generators.append(noise)
        populations.append(pop)
       
        
    
    blob_pop = blob(n=800,plot=False,m=m)
    
    
    for idx in range(len(populations)-1):
        nest.Connect(
            populations[idx],
            populations[idx+1],
            conn_spec=conn_dict_layer,
            syn_spec=syn_dict_layer
        )
    
    
    nest.Connect(populations[-1], blob_pop, conn_dict_ex, syn_spec_ex)
    nest.Connect(populations[-1], blob_pop, conn_dict_inh, syn_spec_inh)
    
    return poisson_generators, noise_generators, populations, blob_pop


def input_to_receptive_field(image_array, poisson_generators, max_rate=200.0):
    if image_array.size != len(poisson_generators):
        print("AMOUNT OF POISSON_GENERATORS DOES NOT MATCH THE ONE OF IMAGE_ARRAYS!!!11\n\n")
        pass

    flat_image = image_array.flatten()
    scaled_rates = (flat_image / 255.0) * max_rate

    for i, rate in enumerate(scaled_rates):
        nest.SetStatus([poisson_generators[i]], {'rate': float(rate)})


try:
    with open("functional_models.json") as f:
        VALID_MODEL_PARAMS = json.load(f)
except FileNotFoundError:
    print(" Warning: functional_models.json not found - parameter validation disabled")
    VALID_MODEL_PARAMS = {}

def filter_params_for_model(model_name, params):
    if not params:
        return {}
    
    if model_name not in VALID_MODEL_PARAMS:
        return {}
    
    model_def = VALID_MODEL_PARAMS[model_name]
    filtered = {}
    
    for key, value in params.items():
        if key in model_def:
            param_info = model_def[key]
            expected_type = param_info.get('type', 'float')
            
            if expected_type == 'array' and isinstance(value, (int, float)):
                print(f" Auto-fixing param '{key}' for {model_name}: {value} -> [{value}]")
                filtered[key] = [float(value)]
            
            elif expected_type == 'float' and isinstance(value, (list, tuple, np.ndarray)):
                if len(value) == 1:
                    print(f" Auto-fixing param '{key}' for {model_name}: {value} -> {value[0]}")
                    filtered[key] = float(value[0])
                else:
                    print(f"Cannot auto-fix param '{key}' for {model_name}: Expected float, got vector {value}")
                    continue
            else:
                filtered[key] = value
    
    return filtered


def clusters_to_neurons(positions, neuron_models, params_per_pop=None, set_positions=True):
    
    populations = []
    
    for i, cluster in enumerate(positions):
        if len(cluster) == 0:
            print(f"  Pop {i}: Leer - übersprungen")
            populations.append([])
            continue
        
        model = neuron_models[i] if i < len(neuron_models) else neuron_models[0]
        
        if params_per_pop and i < len(params_per_pop):
            raw_params = params_per_pop[i].copy() if params_per_pop[i] else {}
            
            validated_params = filter_params_for_model(model, raw_params)
            
            print(f"  Pop {i}: Erstelle {len(cluster)} × {model}")
            print(f"    → {len(raw_params)} Parameter geliefert, {len(validated_params)} gültig")
        else:
            validated_params = {}
            print(f"  Pop {i}: Erstelle {len(cluster)} × {model} mit NEST defaults")
        
        try:
            if set_positions:
                cluster_list = cluster.tolist() if isinstance(cluster, np.ndarray) else cluster
                positions=[]
                if len(cluster) == 1:
                    positions = nest.spatial.free(pos=cluster_list, extent=[1.0, 1.0, 1.0])
                else:
                    positions = nest.spatial.free(pos=cluster_list)
                pop = nest.Create(
                    model,
                    positions=positions,
                    params=validated_params
                )
                
                print(f"    ✓ {len(cluster)} spatial neurons erstellt")
                
                try:
                    actual_pos = nest.GetPosition(pop)
                    if len(actual_pos) == len(cluster):
                        print(f"    ✓ Positionen verifiziert ({len(actual_pos)} Neuronen)")
                    else:
                        print(f"  Position mismatch: {len(actual_pos)} vs {len(cluster)}")
                except Exception as e:
                    print(f"  Position verification failed: {e}")
            else:
                pop = nest.Create(model, len(cluster), params=validated_params)
                print(f"    ✓ {len(cluster)} non-spatial neurons erstellt")
            
            populations.append(pop)
            
        except Exception as e:
            print(f"    ✗ FEHLER bei Pop {i} ({model}): {e}")
            print(f"       Versuchte Parameter: {list(validated_params.keys())}")
            populations.append([])
    
    return populations

def xy_distance(a, b):
    return np.linalg.norm(a[:2] - b[:2])

def eye_lgn_layer(gsl = 16,plot=False):
    eye_layer_size = [gsl,gsl,gsl,gsl,gsl,gsl,gsl,gsl]

    lgn_layer_size = [gsl,gsl,gsl,gsl,gsl]


    eye_1 = create_Grid(m = np.array([1,gsl,10]),grid_size_list = eye_layer_size, plot = False, rot_phi = 0)
    eye_2 = create_Grid(m = np.array([1,-2*gsl,10]),grid_size_list = eye_layer_size, plot = False, rot_phi = 0)


    LGN_1 = create_Grid(m = np.array([20,gsl,15]),grid_size_list = lgn_layer_size, plot = False, rot_phi = 0)
    LGN_2 = create_Grid(m = np.array([20,-2*gsl,15]),grid_size_list = lgn_layer_size, plot = False, rot_phi = 0)


    V1_projection = create_Grid(m = np.array([10,0,0]),grid_size_list=[gsl],rot_theta=0.0,rot_phi=90)


    if plot:
        eyes = eye_1 + eye_2
        plot_point_clusters(eyes,marker_size = 10, alpha = 0.5,linewidths=0.1)
        plot_point_clusters_normalized(eyes,marker_size = 2, alpha = 0.5,linewidths=0.1)
        lgns = LGN_1 + LGN_2
        plot_point_clusters(lgns,marker_size = 10, alpha = 0.5,linewidths=0.1)
        plot_point_clusters_normalized(lgns,marker_size = 2, alpha = 0.5,linewidths=0.1)
    return [eye_1,eye_2,LGN_1,LGN_2,V1_projection]


def vis_cortex_pos(
    gsl          = 16,
    col_height   = 2.0,
    cells_per_col= 140,
    rot_xy       = (0.0, 90.0),
    grid_size        = None,
    rot_theta        = 0,
    rot_phi          = 0,
    transform_matrix = np.diag([1,1,1]),
    dt               = 0.01,
    area_specs = None,
    flow_functions = None,
    experiments = None,
    plot=False):

    if grid_size is None:
        grid_size=np.array([gsl,gsl,gsl])
    if area_specs is None:
        area_specs = {
            "V1": dict(offset_x=10, inner=0.10, outer=0.15),
            "V2": dict(offset_x= 8, inner=0.15, outer=0.20),
            "V3": dict(offset_x= 6, inner=0.20, outer=0.30),
            "V4": dict(offset_x= 4, inner=0.30, outer=0.40),
            "V5": dict(offset_x= 2, inner=0.40, outer=0.45),
            "V6": dict(offset_x= 0, inner=0.45, outer=0.50),
        }
    if experiments is None:
        experiments = {
            "V1C": dict(
                m               = np.array([0,0,-9]),
                wave_params     = dict(sparse_holes=0,
                                       sparsity_factor=0.8,
                                       probability_vector=[0.35,0.25,0.10,0.05,0.20,0.05]),
                types           = [0,1,2,3,4,5],
                num_steps       = 10,
                title           = "V1 computational body"
            ),

            "V2C": dict(
                m               = np.array([0,0,-7]),
                wave_params     = dict(sparse_holes=0,
                                       sparsity_factor=0.8,
                                       probability_vector=[0.30,0.15,0.20,0.05,0.25]),
                types           = [0,1,2,3,4],
                num_steps       = 10,
                title           = "V2 computational body"
            ),
                "V3C": dict(
                m               = np.array([0,0,-5]),
                wave_params     = dict(sparse_holes=0,
                                       sparsity_factor=0.8,
                                       probability_vector=[0.3,0.15,0.15,0.05,0.2]),
                types           = [0,1,2,3,4],
                num_steps       = 10,
                title           = "V3 computational body"
            ),
                "V4C": dict(
                m               = np.array([0,0,-3]),
                wave_params     = dict(sparse_holes=0,
                                       sparsity_factor=0.8,
                                       probability_vector=[0.25,0.25,0.1,0.05,0.15,0.1]),
                types           = [0,1,2,3,4,5],
                num_steps       = 10,
                title           = "V4 computational body"
            ),
                "V5C": dict(
                m               = np.array([0,0,-1]),
                wave_params     = dict(sparse_holes=0,
                                       sparsity_factor=0.8,
                                       probability_vector=[0.45,0.25,0.1,0.1,0.1]),
                types           = [0,1,2,3,4],
                num_steps       = 10,
                title           = "V5 computational body"
            ),
                "V6C": dict(
                m               = np.array([0,0,1]),
                wave_params     = dict(sparse_holes=0,
                                       sparsity_factor=0.8,
                                       probability_vector=[0.35,0.25,0.1,0.05,0.2,0.05]),
                types           = [0,1,2,3,4,5],
                num_steps       = 10,
                title           = "V6 computational body"
            )
        }


    if flow_functions is None:
        f1 = lambda x,y,z : 1.0/(0.3*x**2 + 0.3*y**2 + 0.15)
        flow_functions = [(lambda x,y,z:x,
                           lambda x,y,z:y,
                           f1)] * 6


    results = {}
    all_columns  = {}


    for name, spec in area_specs.items():

        grid_layers   = create_Grid(
            m           = np.array([spec["offset_x"], 0, 0]),
            grid_size_list=[gsl],
            rot_theta   = rot_xy[0],
            rot_phi     = rot_xy[1],
            plot        = False
        )
        projection_xy = grid_layers[0]

        columns = []
        for center in projection_xy:
            col_pts = create_cone(
                m             = center,
                n             = cells_per_col,
                inner_radius  = spec["inner"],
                outer_radius  = spec["outer"],
                height        = col_height,
                rot_theta     = 0.0,
                rot_phi       = 0.0,
                plot          = False
            )
            columns.append(col_pts)
        all_columns[name] = columns


    for name, cfg in experiments.items():
        res = cluster_and_flow(
            grid_size      = grid_size,
            m              = cfg["m"],
            rot_theta      = rot_theta,
            rot_phi        = rot_phi,
            transform_matrix = transform_matrix,
            wave_params    = cfg["wave_params"],
            types          = cfg["types"],
            flow_functions = flow_functions,
            dt             = dt,
            num_steps      = cfg["num_steps"],
            plot_clusters  = plot,
            title          = cfg["title"]
        )
        results[name]=res


    if plot:
        for name, cols in all_columns.items():
            plot_point_clusters(cols,
                                marker_size = 2,
                                alpha       = 0.6,
                                linewidths  = 0.1,
                                title       = f"{name}-Säulen ({len(cols)} Stück)")
    return results, all_columns


def vis2neurons(
    Vn_pop,
    area2models=None,
    plot=False
    ):
    
    model_alias = {"iaf_cond_beta_gap": "iaf_cond_alpha"}
    if(area2models is None):
        area2models = {
            "V1": ["iaf_cond_alpha","aeif_cond_alpha","iaf_cond_exp",
                   "hh_psc_alpha","iaf_cond_alpha","iaf_cond_beta_gap"],
            "V2": ["aeif_cond_alpha","iaf_cond_exp","aeif_cond_alpha",
                   "hh_psc_alpha","iaf_cond_alpha"],
            "V3": ["aeif_cond_alpha","iaf_cond_exp","aeif_cond_alpha",
                   "hh_psc_alpha","iaf_cond_alpha"],
            "V4": ["aeif_cond_alpha","iaf_cond_exp","hh_psc_alpha",
                   "gif_cond_exp","iaf_cond_alpha","iaf_cond_beta_gap"],
            "V5": ["iaf_cond_alpha","aeif_cond_alpha","hh_psc_alpha",
                   "izhikevich","iaf_cond_beta_gap"],
            "V6": ["iaf_cond_alpha","aeif_cond_alpha","hh_psc_alpha",
                   "iaf_psc_delta","iaf_cond_alpha","iaf_cond_beta_gap"]
        }
    Vn_neurons = {}
    for exp_name, clusters in Vn_pop.items():
        models = [model_alias.get(m, m) for m in area2models[exp_name[:2]]]
        models = (models * ((len(clusters) + len(models) - 1) // len(models)))[:len(clusters)]
        Vn_neurons[exp_name] = clusters_to_neurons(clusters, models)
    if(plot):
        for exp_name, cluster_pops in Vn_neurons.items():
            for pop in cluster_pops:
                nest.PlotLayer(pop)
    return Vn_neurons


node_parameters = {
    "types": [0, 1, 2],
    "neuron_models": ["iaf_psc_alpha", "iaf_psc_exp", "iaf_psc_alpha"],
    "grid_size": [10, 10, 10],
    "m": [0.0, 0.0, 0.0],
    "rot_theta": 0.0,
    "rot_phi": 0.0,
    "transform_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "dt": 0.01,
    "old": True,
    "num_steps": 8,
    "plot_clusters": True,
    "title": "3 populations",
    "sparse_holes": 0,
    "sparsity_factor": 0.9,
    "probability_vector": [0.3, 0.2, 0.4],
    "name": "TestNode",
    "id": 0,
    "distribution": [],
    "conn_prob": [],
    "polynom_max_power": 5,
    "coefficients": None,
    "center_of_mass": np.array((0.0, 0.0, 0.0)),
    "displacement": np.array((0.0, 0.0, 0.0)),
    "displacement_factor": 1.0,
    "field": None,
    
    "encoded_polynoms_per_type": [
        [
            {'indices': [[0, 1], [0, 0]], 'coefficients': [1.5, 0.5], 'n': 5, 'decay': 0.5},
            {'indices': [[1, 1], [1, 0]], 'coefficients': [1.0, 0.3], 'n': 5, 'decay': 0.5},
            {'indices': [[2, 1], [2, 0]], 'coefficients': [0.8, 0.2], 'n': 5, 'decay': 0.5}
        ],
        [
            {'indices': [[0, 1]], 'coefficients': [1.2], 'n': 5, 'decay': 0.5},
            {'indices': [[1, 1]], 'coefficients': [0.9], 'n': 5, 'decay': 0.5},
            {'indices': [[2, 1]], 'coefficients': [0.7], 'n': 5, 'decay': 0.5}
        ],
        [
            {'indices': [[0, 1]], 'coefficients': [0.6], 'n': 5, 'decay': 0.5},
            {'indices': [[1, 1]], 'coefficients': [1.1], 'n': 5, 'decay': 0.5},
            {'indices': [[2, 1]], 'coefficients': [0.9], 'n': 5, 'decay': 0.5}
        ]
    ]
}


neuron_colors = {
    "aeif_cond_alpha": "#1E88E5",
    "aeif_cond_alpha_multisynapse": "#1976D2",
    "aeif_cond_beta_multisynapse": "#1565C0",
    "aeif_cond_exp": "#0D47A1",
    "aeif_psc_alpha": "#42A5F5",
    "aeif_psc_delta": "#64B5F6",
    "aeif_psc_delta_clopath": "#90CAF9",
    "aeif_psc_exp": "#2196F3",
    
    "iaf_cond_alpha": "#43A047",
    "iaf_cond_beta": "#388E3C",
    "iaf_cond_exp": "#2E7D32",
    "iaf_cond_exp_sfa_rr": "#1B5E20",
    "iaf_cond_alpha_mc": "#66BB6A",
    "iaf_psc_alpha": "#4CAF50",
    "iaf_psc_alpha_multisynapse": "#81C784",
    "iaf_psc_delta": "#A5D6A7",
    "iaf_psc_exp": "#66BB6A",
    "iaf_psc_exp_multisynapse": "#8BC34A",
    "iaf_tum_2000": "#9CCC65",
    "iaf_chs_2007": "#AED581",
    "iaf_chxk_2008": "#C5E1A5",
    "iaf_psc_alpha_ps": "#558B2F",
    "iaf_psc_delta_ps": "#689F38",
    "iaf_psc_exp_htum": "#7CB342",
    "iaf_psc_exp_ps": "#8BC34A",
    "iaf_psc_exp_ps_lossless": "#9CCC65",
    
    "hh_cond_exp_traub": "#D32F2F",
    "hh_cond_beta_gap_traub": "#C62828",
    "hh_psc_alpha": "#B71C1C",
    "hh_psc_alpha_clopath": "#E53935",
    "hh_psc_alpha_gap": "#F44336",
    
    "gif_cond_exp": "#7B1FA2",
    "gif_cond_exp_multisynapse": "#6A1B9A",
    "gif_psc_exp": "#4A148C",
    "gif_psc_exp_multisynapse": "#8E24AA",
    "gif_pop_psc_exp": "#9C27B0",
    
    "mat2_psc_exp": "#F57C00",
    "amat2_psc_exp": "#EF6C00",
    
    "glif_cond": "#00ACC1",
    "glif_psc": "#0097A7",
    
    "izhikevich": "#FFB300",
    "pp_psc_delta": "#5E35B1",
    "pp_cond_exp_mc_urbanczik": "#3949AB",
    "parrot_neuron": "#78909C",
    "parrot_neuron_ps": "#90A4AE",
    "mcculloch_pitts_neuron": "#455A64",
    "siegert_neuron": "#00897B",
    "ht_neuron": "#D81B60",
    "erfc_neuron": "#6D4C41",
    "ginzburg_neuron": "#8D6E63",
}
raw_models = list(neuron_colors.keys())
successful_neuron_models = sorted(raw_models, key=lambda x: x.lower())


region_names = {
    'Neocortex Layer 2/3': 'Neocortex_L23',
    'Neocortex Layer 4': 'Neocortex_L4',
    'Neocortex Layer 5': 'Neocortex_L5',
    'Neocortex Layer 6': 'Neocortex_L6',
    'Hippocampus CA1': 'Hippocampus_CA1',
    'Hippocampus CA3': 'Hippocampus_CA3',
    'Thalamus Relay': 'Thalamus_Relay',
    'Thalamus Reticular Nucleus': 'Thalamus_Reticular'
}

distributions = {
    'aeif_cond_alpha': {'Neocortex_L23': 0.50, 'Neocortex_L4': 0.45, 'Neocortex_L5': 0.55, 'Neocortex_L6': 0.50, 'Hippocampus_CA1': 0.55, 'Hippocampus_CA3': 0.55, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.02},
    'aeif_cond_alpha_multisynapse': {'Neocortex_L23': 0.55, 'Neocortex_L4': 0.50, 'Neocortex_L5': 0.60, 'Neocortex_L6': 0.55, 'Hippocampus_CA1': 0.60, 'Hippocampus_CA3': 0.60, 'Thalamus_Relay': 0.08, 'Thalamus_Reticular': 0.03},
    'aeif_cond_beta_multisynapse': {'Neocortex_L23': 0.55, 'Neocortex_L4': 0.50, 'Neocortex_L5': 0.58, 'Neocortex_L6': 0.55, 'Hippocampus_CA1': 0.58, 'Hippocampus_CA3': 0.58, 'Thalamus_Relay': 0.08, 'Thalamus_Reticular': 0.03},
    'aeif_cond_exp': {'Neocortex_L23': 0.50, 'Neocortex_L4': 0.45, 'Neocortex_L5': 0.55, 'Neocortex_L6': 0.50, 'Hippocampus_CA1': 0.55, 'Hippocampus_CA3': 0.55, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.02},
    'aeif_psc_alpha': {'Neocortex_L23': 0.50, 'Neocortex_L4': 0.45, 'Neocortex_L5': 0.55, 'Neocortex_L6': 0.50, 'Hippocampus_CA1': 0.55, 'Hippocampus_CA3': 0.55, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.02},
    'aeif_psc_delta': {'Neocortex_L23': 0.48, 'Neocortex_L4': 0.43, 'Neocortex_L5': 0.53, 'Neocortex_L6': 0.48, 'Hippocampus_CA1': 0.53, 'Hippocampus_CA3': 0.53, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.02},
    'aeif_psc_delta_clopath': {'Neocortex_L23': 0.60, 'Neocortex_L4': 0.45, 'Neocortex_L5': 0.65, 'Neocortex_L6': 0.55, 'Hippocampus_CA1': 0.55, 'Hippocampus_CA3': 0.50, 'Thalamus_Relay': 0.03, 'Thalamus_Reticular': 0.02},
    'aeif_psc_exp': {'Neocortex_L23': 0.50, 'Neocortex_L4': 0.45, 'Neocortex_L5': 0.55, 'Neocortex_L6': 0.50, 'Hippocampus_CA1': 0.55, 'Hippocampus_CA3': 0.55, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.02},
    'amat2_psc_exp': {'Neocortex_L23': 0.55, 'Neocortex_L4': 0.50, 'Neocortex_L5': 0.60, 'Neocortex_L6': 0.55, 'Hippocampus_CA1': 0.58, 'Hippocampus_CA3': 0.58, 'Thalamus_Relay': 0.08, 'Thalamus_Reticular': 0.03},
    'gif_cond_exp': {'Neocortex_L23': 0.55, 'Neocortex_L4': 0.50, 'Neocortex_L5': 0.70, 'Neocortex_L6': 0.55, 'Hippocampus_CA1': 0.58, 'Hippocampus_CA3': 0.55, 'Thalamus_Relay': 0.08, 'Thalamus_Reticular': 0.05},
    'hh_cond_beta_gap_traub': {'Neocortex_L23': 0.10, 'Neocortex_L4': 0.08, 'Neocortex_L5': 0.12, 'Neocortex_L6': 0.10, 'Hippocampus_CA1': 0.25, 'Hippocampus_CA3': 0.75, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.15},
    'hh_psc_alpha': {'Neocortex_L23': 0.45, 'Neocortex_L4': 0.40, 'Neocortex_L5': 0.50, 'Neocortex_L6': 0.45, 'Hippocampus_CA1': 0.60, 'Hippocampus_CA3': 0.60, 'Thalamus_Relay': 0.10, 'Thalamus_Reticular': 0.05},
    'hh_psc_alpha_clopath': {'Neocortex_L23': 0.55, 'Neocortex_L4': 0.40, 'Neocortex_L5': 0.60, 'Neocortex_L6': 0.50, 'Hippocampus_CA1': 0.55, 'Hippocampus_CA3': 0.50, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.03},
    'hh_psc_alpha_gap': {'Neocortex_L23': 0.15, 'Neocortex_L4': 0.15, 'Neocortex_L5': 0.18, 'Neocortex_L6': 0.15, 'Hippocampus_CA1': 0.30, 'Hippocampus_CA3': 0.70, 'Thalamus_Relay': 0.08, 'Thalamus_Reticular': 0.25},
    'ht_neuron': {'Neocortex_L23': 0.05, 'Neocortex_L4': 0.08, 'Neocortex_L5': 0.05, 'Neocortex_L6': 0.10, 'Hippocampus_CA1': 0.03, 'Hippocampus_CA3': 0.03, 'Thalamus_Relay': 0.90, 'Thalamus_Reticular': 0.15},
    'iaf_chs_2007': {'Neocortex_L23': 0.02, 'Neocortex_L4': 0.05, 'Neocortex_L5': 0.02, 'Neocortex_L6': 0.03, 'Hippocampus_CA1': 0.02, 'Hippocampus_CA3': 0.02, 'Thalamus_Relay': 0.95, 'Thalamus_Reticular': 0.05},
    'iaf_chxk_2008': {'Neocortex_L23': 0.02, 'Neocortex_L4': 0.05, 'Neocortex_L5': 0.02, 'Neocortex_L6': 0.03, 'Hippocampus_CA1': 0.02, 'Hippocampus_CA3': 0.02, 'Thalamus_Relay': 0.95, 'Thalamus_Reticular': 0.05},
    'iaf_cond_alpha': {'Neocortex_L23': 0.65, 'Neocortex_L4': 0.60, 'Neocortex_L5': 0.65, 'Neocortex_L6': 0.65, 'Hippocampus_CA1': 0.70, 'Hippocampus_CA3': 0.70, 'Thalamus_Relay': 0.60, 'Thalamus_Reticular': 0.15},
    'iaf_cond_alpha_mc': {'Neocortex_L23': 0.70, 'Neocortex_L4': 0.60, 'Neocortex_L5': 0.75, 'Neocortex_L6': 0.70, 'Hippocampus_CA1': 0.75, 'Hippocampus_CA3': 0.70, 'Thalamus_Relay': 0.10, 'Thalamus_Reticular': 0.05},
    'iaf_cond_beta': {'Neocortex_L23': 0.65, 'Neocortex_L4': 0.60, 'Neocortex_L5': 0.65, 'Neocortex_L6': 0.65, 'Hippocampus_CA1': 0.70, 'Hippocampus_CA3': 0.70, 'Thalamus_Relay': 0.60, 'Thalamus_Reticular': 0.15},
    'iaf_cond_exp': {'Neocortex_L23': 0.70, 'Neocortex_L4': 0.65, 'Neocortex_L5': 0.70, 'Neocortex_L6': 0.70, 'Hippocampus_CA1': 0.75, 'Hippocampus_CA3': 0.75, 'Thalamus_Relay': 0.65, 'Thalamus_Reticular': 0.18},
    'iaf_cond_exp_sfa_rr': {'Neocortex_L23': 0.35, 'Neocortex_L4': 0.25, 'Neocortex_L5': 0.38, 'Neocortex_L6': 0.35, 'Hippocampus_CA1': 0.35, 'Hippocampus_CA3': 0.35, 'Thalamus_Relay': 0.10, 'Thalamus_Reticular': 0.05},
    'iaf_psc_alpha': {'Neocortex_L23': 0.70, 'Neocortex_L4': 0.65, 'Neocortex_L5': 0.70, 'Neocortex_L6': 0.70, 'Hippocampus_CA1': 0.75, 'Hippocampus_CA3': 0.75, 'Thalamus_Relay': 0.65, 'Thalamus_Reticular': 0.18},
    'iaf_psc_alpha_multisynapse': {'Neocortex_L23': 0.72, 'Neocortex_L4': 0.67, 'Neocortex_L5': 0.72, 'Neocortex_L6': 0.72, 'Hippocampus_CA1': 0.77, 'Hippocampus_CA3': 0.77, 'Thalamus_Relay': 0.67, 'Thalamus_Reticular': 0.20},
    'iaf_psc_delta': {'Neocortex_L23': 0.68, 'Neocortex_L4': 0.63, 'Neocortex_L5': 0.68, 'Neocortex_L6': 0.68, 'Hippocampus_CA1': 0.73, 'Hippocampus_CA3': 0.73, 'Thalamus_Relay': 0.63, 'Thalamus_Reticular': 0.17},
    'iaf_psc_exp': {'Neocortex_L23': 0.75, 'Neocortex_L4': 0.70, 'Neocortex_L5': 0.75, 'Neocortex_L6': 0.75, 'Hippocampus_CA1': 0.78, 'Hippocampus_CA3': 0.78, 'Thalamus_Relay': 0.70, 'Thalamus_Reticular': 0.20},
    'iaf_psc_exp_multisynapse': {'Neocortex_L23': 0.75, 'Neocortex_L4': 0.70, 'Neocortex_L5': 0.75, 'Neocortex_L6': 0.75, 'Hippocampus_CA1': 0.78, 'Hippocampus_CA3': 0.78, 'Thalamus_Relay': 0.70, 'Thalamus_Reticular': 0.20},
    'iaf_tum_2000': {'Neocortex_L23': 0.70, 'Neocortex_L4': 0.65, 'Neocortex_L5': 0.70, 'Neocortex_L6': 0.68, 'Hippocampus_CA1': 0.70, 'Hippocampus_CA3': 0.72, 'Thalamus_Relay': 0.55, 'Thalamus_Reticular': 0.15},
    'izhikevich': {'Neocortex_L23': 0.80, 'Neocortex_L4': 0.75, 'Neocortex_L5': 0.80, 'Neocortex_L6': 0.78, 'Hippocampus_CA1': 0.80, 'Hippocampus_CA3': 0.78, 'Thalamus_Relay': 0.60, 'Thalamus_Reticular': 0.25},
    'mat2_psc_exp': {'Neocortex_L23': 0.60, 'Neocortex_L4': 0.55, 'Neocortex_L5': 0.65, 'Neocortex_L6': 0.60, 'Hippocampus_CA1': 0.62, 'Hippocampus_CA3': 0.62, 'Thalamus_Relay': 0.12, 'Thalamus_Reticular': 0.08},
    'mcculloch_pitts_neuron': {'Neocortex_L23': 0.05, 'Neocortex_L4': 0.05, 'Neocortex_L5': 0.05, 'Neocortex_L6': 0.05, 'Hippocampus_CA1': 0.05, 'Hippocampus_CA3': 0.05, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.05},
    'parrot_neuron': {'Neocortex_L23': 0.01, 'Neocortex_L4': 0.01, 'Neocortex_L5': 0.01, 'Neocortex_L6': 0.01, 'Hippocampus_CA1': 0.01, 'Hippocampus_CA3': 0.01, 'Thalamus_Relay': 0.01, 'Thalamus_Reticular': 0.01},
    'pp_psc_delta': {'Neocortex_L23': 0.60, 'Neocortex_L4': 0.55, 'Neocortex_L5': 0.62, 'Neocortex_L6': 0.60, 'Hippocampus_CA1': 0.65, 'Hippocampus_CA3': 0.65, 'Thalamus_Relay': 0.08, 'Thalamus_Reticular': 0.05},
    'siegert_neuron': {'Neocortex_L23': 0.80, 'Neocortex_L4': 0.75, 'Neocortex_L5': 0.80, 'Neocortex_L6': 0.80, 'Hippocampus_CA1': 0.85, 'Hippocampus_CA3': 0.85, 'Thalamus_Relay': 0.75, 'Thalamus_Reticular': 0.20}
}
distributions2 = {
    'aeif_cond_alpha': {
        'Neocortex_L23': 0.50, 'Neocortex_L4': 0.45, 'Neocortex_L5': 0.55, 'Neocortex_L6': 0.50,
        'Hippocampus_CA1': 0.55, 'Hippocampus_CA3': 0.55, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.02,
        'BasalGanglia_STN': 0.75,
        'BasalGanglia_GPe_Prototypical': 0.65,
        'Amygdala_BLA': 0.60,
        'Cerebellum_Purkinje': 0.40,
        'Brainstem_Raphe': 0.55,
        'BasalGanglia_SNr': 0.60
    },

    'aeif_cond_alpha_multisynapse': {
        'Neocortex_L23': 0.55, 'Neocortex_L4': 0.50, 'Neocortex_L5': 0.60, 'Neocortex_L6': 0.55,
        'Hippocampus_CA1': 0.60, 'Hippocampus_CA3': 0.60, 'Thalamus_Relay': 0.08, 'Thalamus_Reticular': 0.03,
        'BasalGanglia_Striatum_FSI': 0.70,
        'Cerebellum_Purkinje': 0.85,
        'Amygdala_CeA': 0.65,
        'Olfactory_Mitral': 0.75,
        'Amygdala_ITC': 0.70
    },

    'aeif_cond_beta_multisynapse': {
        'Neocortex_L23': 0.55, 'Neocortex_L4': 0.50, 'Neocortex_L5': 0.58, 'Neocortex_L6': 0.55,
        'Hippocampus_CA1': 0.58, 'Hippocampus_CA3': 0.58, 'Thalamus_Relay': 0.08, 'Thalamus_Reticular': 0.03,
        'BasalGanglia_Striatum_MSN_D1': 0.65,
        'BasalGanglia_Striatum_MSN_D2': 0.65,
        'Hippocampus_DG_Granule': 0.50,
        'BasalGanglia_SNc': 0.45
    },

    'aeif_cond_exp': {
        'Neocortex_L23': 0.50, 'Neocortex_L4': 0.45, 'Neocortex_L5': 0.55, 'Neocortex_L6': 0.50,
        'Hippocampus_CA1': 0.55, 'Hippocampus_CA3': 0.55, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.02,
        'BasalGanglia_GPe_Prototypical': 0.80,
        'BasalGanglia_GPe_Arkypallidal': 0.75,
        'Cerebellum_IO': 0.70,
        'Spinal_Motor': 0.60,
        'Retina_Ganglion': 0.65,
        'Hippocampus_DG_Mossy': 0.55
    },

    'aeif_psc_alpha': {
        'Neocortex_L23': 0.50, 'Neocortex_L4': 0.45, 'Neocortex_L5': 0.55, 'Neocortex_L6': 0.50,
        'Hippocampus_CA1': 0.55, 'Hippocampus_CA3': 0.55, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.02,
        'BasalGanglia_STN': 0.60,
        'Brainstem_LocusCoeruleus': 0.50,
        'Cerebellum_Stellate': 0.45,
        'Cerebellum_Basket': 0.45
    },

    'aeif_psc_delta': {
        'Neocortex_L23': 0.48, 'Neocortex_L4': 0.43, 'Neocortex_L5': 0.53, 'Neocortex_L6': 0.48,
        'Hippocampus_CA1': 0.53, 'Hippocampus_CA3': 0.53, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.02,
        'Cerebellum_Granule': 0.90,
        'Olfactory_Granule': 0.80,
        'BasalGanglia_Striatum_FSI': 0.40,
        'Spinal_Renshaw': 0.70
    },

    'aeif_psc_delta_clopath': {
        'Neocortex_L23': 0.60, 'Neocortex_L4': 0.45, 'Neocortex_L5': 0.65, 'Neocortex_L6': 0.55,
        'Hippocampus_CA1': 0.55, 'Hippocampus_CA3': 0.50, 'Thalamus_Relay': 0.03, 'Thalamus_Reticular': 0.02,
        'BasalGanglia_Striatum_MSN_D1': 0.50,
        'Amygdala_BLA': 0.55,
        'Cerebellum_Purkinje': 0.30
    },

    'aeif_psc_exp': {
        'Neocortex_L23': 0.50, 'Neocortex_L4': 0.45, 'Neocortex_L5': 0.55, 'Neocortex_L6': 0.50,
        'Hippocampus_CA1': 0.55, 'Hippocampus_CA3': 0.55, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.02,
        'BasalGanglia_GPe_Prototypical': 0.50,
        'Cerebellum_DCN': 0.60,
        'Brainstem_Raphe': 0.50
    },

    'aeif_cond_exp_sfa_rr': {
         'Neocortex_L23': 0.35, 'Neocortex_L4': 0.25, 'Neocortex_L5': 0.38, 'Neocortex_L6': 0.35,
         'Hippocampus_CA1': 0.35, 'Hippocampus_CA3': 0.35, 'Thalamus_Relay': 0.10, 'Thalamus_Reticular': 0.05,
         'Spinal_Motor': 0.95,
         'Brainstem_LocusCoeruleus': 0.70,
         'Retina_Ganglion': 0.60
    },

    'amat2_psc_exp': {
        'Neocortex_L23': 0.55, 'Neocortex_L4': 0.50, 'Neocortex_L5': 0.60, 'Neocortex_L6': 0.55,
        'Hippocampus_CA1': 0.58, 'Hippocampus_CA3': 0.58, 'Thalamus_Relay': 0.08, 'Thalamus_Reticular': 0.03,
        'Cerebellum_Granule': 0.40,
        'BasalGanglia_Striatum_MSN_D1': 0.30
    },

    'gif_cond_exp': {
        'Neocortex_L23': 0.55, 'Neocortex_L4': 0.50, 'Neocortex_L5': 0.70, 'Neocortex_L6': 0.55,
        'Hippocampus_CA1': 0.58, 'Hippocampus_CA3': 0.55, 'Thalamus_Relay': 0.08, 'Thalamus_Reticular': 0.05,
        'Retina_Ganglion': 0.80,
        'BasalGanglia_GPe_Arkypallidal': 0.70,
        'Olfactory_Tufted': 0.60
    },

    'hh_cond_beta_gap_traub': {
        'Neocortex_L23': 0.10, 'Neocortex_L4': 0.08, 'Neocortex_L5': 0.12, 'Neocortex_L6': 0.10,
        'Hippocampus_CA1': 0.25, 'Hippocampus_CA3': 0.75, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.15,
        'Cerebellum_Purkinje': 0.35,
        'Olfactory_Mitral': 0.60,
        'BasalGanglia_Striatum_FSI': 0.50,
        'Brainstem_LocusCoeruleus': 0.65
    },

    'hh_psc_alpha': {
        'Neocortex_L23': 0.45, 'Neocortex_L4': 0.40, 'Neocortex_L5': 0.50, 'Neocortex_L6': 0.45,
        'Hippocampus_CA1': 0.60, 'Hippocampus_CA3': 0.60, 'Thalamus_Relay': 0.10, 'Thalamus_Reticular': 0.05,
        'BasalGanglia_SNc': 0.40,
        'Spinal_Renshaw': 0.55,
        'Cerebellum_Golgi': 0.50
    },

    'hh_psc_alpha_clopath': {
        'Neocortex_L23': 0.55, 'Neocortex_L4': 0.40, 'Neocortex_L5': 0.60, 'Neocortex_L6': 0.50,
        'Hippocampus_CA1': 0.55, 'Hippocampus_CA3': 0.50, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.03,
        'Amygdala_BLA': 0.50,
        'BasalGanglia_Striatum_MSN_D1': 0.45
    },

    'hh_psc_alpha_gap': {
        'Neocortex_L23': 0.15, 'Neocortex_L4': 0.15, 'Neocortex_L5': 0.18, 'Neocortex_L6': 0.15,
        'Hippocampus_CA1': 0.30, 'Hippocampus_CA3': 0.70, 'Thalamus_Relay': 0.08, 'Thalamus_Reticular': 0.25,
        'Cerebellum_Purkinje': 0.40,
        'Olfactory_Mitral': 0.55,
        'BasalGanglia_GPe_Prototypical': 0.20
    },

    'ht_neuron': {
        'Neocortex_L23': 0.05, 'Neocortex_L4': 0.08, 'Neocortex_L5': 0.05, 'Neocortex_L6': 0.10,
        'Hippocampus_CA1': 0.03, 'Hippocampus_CA3': 0.03, 'Thalamus_Relay': 0.90, 'Thalamus_Reticular': 0.15,
        'BasalGanglia_GPe_Prototypical': 0.10,
        'Retina_Ganglion': 0.15
    },

    'izhikevich': {
        'Neocortex_L23': 0.80, 'Neocortex_L4': 0.75, 'Neocortex_L5': 0.80, 'Neocortex_L6': 0.78,
        'Hippocampus_CA1': 0.80, 'Hippocampus_CA3': 0.78, 'Thalamus_Relay': 0.60, 'Thalamus_Reticular': 0.25,
        'BasalGanglia_Striatum_MSN_D1': 0.95,
        'BasalGanglia_Striatum_MSN_D2': 0.95,
        'BasalGanglia_STN': 0.85,
        'BasalGanglia_GPe_Prototypical': 0.85,
        'Amygdala_BLA': 0.80,
        'Cerebellum_IO': 0.90,
        'Hippocampus_DG_Mossy': 0.85
    },

    'mcculloch_pitts_neuron': {
        'Neocortex_L23': 0.05, 'Neocortex_L4': 0.05, 'Neocortex_L5': 0.05, 'Neocortex_L6': 0.05,
        'Hippocampus_CA1': 0.05, 'Hippocampus_CA3': 0.05, 'Thalamus_Relay': 0.05, 'Thalamus_Reticular': 0.05,
        'Cerebellum_Granule': 0.20,
        'BasalGanglia_Striatum_MSN_D1': 0.01
    },

    'parrot_neuron': {
        'Neocortex_L23': 0.01, 'Neocortex_L4': 0.01, 'Neocortex_L5': 0.01, 'Neocortex_L6': 0.01,
        'Hippocampus_CA1': 0.01, 'Hippocampus_CA3': 0.01, 'Thalamus_Relay': 0.01, 'Thalamus_Reticular': 0.01,
        'BasalGanglia_Striatum_MSN_D1': 0.01,
        'Cerebellum_Purkinje': 0.01,
        'Spinal_Motor': 0.01
    },

    'siegert_neuron': {
        'Neocortex_L23': 0.80, 'Neocortex_L4': 0.75, 'Neocortex_L5': 0.80, 'Neocortex_L6': 0.80,
        'Hippocampus_CA1': 0.85, 'Hippocampus_CA3': 0.85, 'Thalamus_Relay': 0.75, 'Thalamus_Reticular': 0.20,
        'BasalGanglia_Striatum_MSN_D1': 0.30,
        'Cerebellum_Granule': 0.70,
        'Retina_Ganglion': 0.50
    },

    'iaf_cond_alpha': {
        'Neocortex_L23': 0.65, 'Neocortex_L4': 0.60, 'Neocortex_L5': 0.65, 'Neocortex_L6': 0.65,
        'Hippocampus_CA1': 0.70, 'Hippocampus_CA3': 0.70, 'Thalamus_Relay': 0.60, 'Thalamus_Reticular': 0.15,
        'Cerebellum_Granule': 0.80,
        'BasalGanglia_Striatum_FSI': 0.30,
        'Spinal_Interneuron': 0.70
    },

    'iaf_cond_alpha_mc': {
        'Neocortex_L23': 0.70, 'Neocortex_L4': 0.60, 'Neocortex_L5': 0.75, 'Neocortex_L6': 0.70,
        'Hippocampus_CA1': 0.75, 'Hippocampus_CA3': 0.70, 'Thalamus_Relay': 0.10, 'Thalamus_Reticular': 0.05,
        'Cerebellum_DCN': 0.50,
        'Brainstem_Raphe': 0.40
    },

    'iaf_cond_beta': {
        'Neocortex_L23': 0.65, 'Neocortex_L4': 0.60, 'Neocortex_L5': 0.65, 'Neocortex_L6': 0.65,
        'Hippocampus_CA1': 0.70, 'Hippocampus_CA3': 0.70, 'Thalamus_Relay': 0.60, 'Thalamus_Reticular': 0.15,
        'BasalGanglia_Striatum_FSI': 0.60,
        'Amygdala_ITC': 0.55
    },

    'iaf_cond_exp': {
        'Neocortex_L23': 0.70, 'Neocortex_L4': 0.65, 'Neocortex_L5': 0.70, 'Neocortex_L6': 0.70,
        'Hippocampus_CA1': 0.75, 'Hippocampus_CA3': 0.75, 'Thalamus_Relay': 0.65, 'Thalamus_Reticular': 0.18,
        'Cerebellum_Granule': 0.85,
        'BasalGanglia_GPe_Prototypical': 0.30
    },

    'iaf_cond_exp_sfa_rr': {
        'Neocortex_L23': 0.35, 'Neocortex_L4': 0.25, 'Neocortex_L5': 0.38, 'Neocortex_L6': 0.35,
        'Hippocampus_CA1': 0.35, 'Hippocampus_CA3': 0.35, 'Thalamus_Relay': 0.10, 'Thalamus_Reticular': 0.05,
        'Spinal_Motor': 0.90,
        'Amygdala_BLA': 0.60
    },

    'iaf_psc_alpha': {
        'Neocortex_L23': 0.70, 'Neocortex_L4': 0.65, 'Neocortex_L5': 0.70, 'Neocortex_L6': 0.70,
        'Hippocampus_CA1': 0.75, 'Hippocampus_CA3': 0.75, 'Thalamus_Relay': 0.65, 'Thalamus_Reticular': 0.18,
        'Cerebellum_Granule': 0.80,
        'Spinal_Interneuron': 0.75
    },

    'iaf_psc_alpha_multisynapse': {
        'Neocortex_L23': 0.72, 'Neocortex_L4': 0.67, 'Neocortex_L5': 0.72, 'Neocortex_L6': 0.72,
        'Hippocampus_CA1': 0.77, 'Hippocampus_CA3': 0.77, 'Thalamus_Relay': 0.67, 'Thalamus_Reticular': 0.20,
        'Olfactory_Mitral': 0.50,
        'Cerebellum_Purkinje': 0.60
    },

    'iaf_psc_delta': {
        'Neocortex_L23': 0.68, 'Neocortex_L4': 0.63, 'Neocortex_L5': 0.68, 'Neocortex_L6': 0.68,
        'Hippocampus_CA1': 0.73, 'Hippocampus_CA3': 0.73, 'Thalamus_Relay': 0.63, 'Thalamus_Reticular': 0.17,
        'Cerebellum_Granule': 0.95,
        'Olfactory_Granule': 0.90
    },

    'iaf_psc_exp': {
        'Neocortex_L23': 0.75, 'Neocortex_L4': 0.70, 'Neocortex_L5': 0.75, 'Neocortex_L6': 0.75,
        'Hippocampus_CA1': 0.78, 'Hippocampus_CA3': 0.78, 'Thalamus_Relay': 0.70, 'Thalamus_Reticular': 0.20,
        'Cerebellum_Granule': 0.85,
        'Spinal_Interneuron': 0.60
    },

    'iaf_psc_exp_multisynapse': {
        'Neocortex_L23': 0.75, 'Neocortex_L4': 0.70, 'Neocortex_L5': 0.75, 'Neocortex_L6': 0.75,
        'Hippocampus_CA1': 0.78, 'Hippocampus_CA3': 0.78, 'Thalamus_Relay': 0.70, 'Thalamus_Reticular': 0.20,
        'Cerebellum_Purkinje': 0.65,
        'Amygdala_CeA': 0.50
    },

    'iaf_tum_2000': {
        'Neocortex_L23': 0.70, 'Neocortex_L4': 0.65, 'Neocortex_L5': 0.70, 'Neocortex_L6': 0.68,
        'Hippocampus_CA1': 0.70, 'Hippocampus_CA3': 0.72, 'Thalamus_Relay': 0.55, 'Thalamus_Reticular': 0.15,
        'Cerebellum_Granule': 0.50,
        'BasalGanglia_Striatum_MSN_D1': 0.10
    },
    
    'iaf_chs_2007': {
        'Neocortex_L23': 0.02, 'Neocortex_L4': 0.05, 'Neocortex_L5': 0.02, 'Neocortex_L6': 0.03,
        'Hippocampus_CA1': 0.02, 'Hippocampus_CA3': 0.02, 'Thalamus_Relay': 0.95, 'Thalamus_Reticular': 0.05,
        'Retina_Ganglion': 0.05,
        'Cerebellum_IO': 0.01
    },

    'iaf_chxk_2008': {
        'Neocortex_L23': 0.02, 'Neocortex_L4': 0.05, 'Neocortex_L5': 0.02, 'Neocortex_L6': 0.03,
        'Hippocampus_CA1': 0.02, 'Hippocampus_CA3': 0.02, 'Thalamus_Relay': 0.95, 'Thalamus_Reticular': 0.05,
        'Retina_Ganglion': 0.05,
        'Cerebellum_IO': 0.01
    },

    'mat2_psc_exp': {
        'Neocortex_L23': 0.60, 'Neocortex_L4': 0.55, 'Neocortex_L5': 0.65, 'Neocortex_L6': 0.60,
        'Hippocampus_CA1': 0.62, 'Hippocampus_CA3': 0.62, 'Thalamus_Relay': 0.12, 'Thalamus_Reticular': 0.08,
        'BasalGanglia_STN': 0.20,
        'Cerebellum_DCN': 0.30
    },

    'pp_psc_delta': {
        'Neocortex_L23': 0.60, 'Neocortex_L4': 0.55, 'Neocortex_L5': 0.62, 'Neocortex_L6': 0.60,
        'Hippocampus_CA1': 0.65, 'Hippocampus_CA3': 0.65, 'Thalamus_Relay': 0.08, 'Thalamus_Reticular': 0.05,
        'Cerebellum_Granule': 0.60,
        'Olfactory_Granule': 0.55
    }
}


def get_probability_vector(region_full_name):
    if region_full_name not in region_names:
        raise ValueError("")
    region_abbrev = region_names[region_full_name]
    raw_values = [distributions[model][region_abbrev] for model in successful_neuron_models]
    raw_array = np.array(raw_values)
    total = np.sum(raw_array)
    if total == 0:
        raise ValueError("")
    return raw_array / total

def rand_prob_vector(num_classes):
    weights = np.random.rand(num_classes)
    prob_random = weights/weights.sum()
    return prob_random

def random_direction(length=1.0):
    v = np.random.randn(3)
    v = v / np.linalg.norm(v)
    return v * length

def create_lambda_array_zero(size):
    rows, cols = size
    return [
        tuple(lambda x, y, z: 0 for _ in range(cols))
        for _ in range(rows)
    ]

def plot_graph_3d(graph,
                  figsize=(12, 10),
                  node_color='red',
                  root_color='green',
                  edge_color='blue',
                  node_size=100,
                  root_size=200,
                  edge_alpha=0.4,
                  show_labels=True,
                  title=None):

    arr = np.array([node.center_of_mass for node in graph.node_list])
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2],
               c=node_color, marker='o', s=node_size, alpha=0.8)
    
    for node in graph.node_list:
        pos_from = np.array(node.center_of_mass)
        for next_node in node.next:
            pos_to = np.array(next_node.center_of_mass)
            ax.plot([pos_from[0], pos_to[0]],
                    [pos_from[1], pos_to[1]],
                    [pos_from[2], pos_to[2]],
                    color=edge_color, alpha=edge_alpha, linewidth=1.5)
    
    if show_labels:
        for node in graph.node_list:
            pos = node.center_of_mass
            ax.text(pos[0], pos[1], pos[2], f'  {node.id}', fontsize=9)
    

    if graph.node_list:
        root_node = graph.node_list[0]
        root_pos = root_node.center_of_mass
        ax.scatter([root_pos[0]], [root_pos[1]], [root_pos[2]],
                   c=root_color, marker='o', s=root_size, alpha=1.0, label='Root')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title is None:
        title = f'3D Graph Structure (nodes={graph.nodes})'
    ax.set_title(title)
    
    ax.legend()
    plt.show()
def get_raw_shape_points(tool_type, params):
    n = int(params.get('n_neurons', 100))
    
    if tool_type == 'CCW':
        r = float(params.get('radius', 5.0))
        return circle3d(n=n, r=r, m=np.zeros(3), plot=False)
        
    elif tool_type == 'Blob':
        r = float(params.get('radius', 5.0))
        return blob_positions(n=n, m=np.zeros(3), r=r, plot=False)
        
    elif tool_type == 'Cone':
        return create_cone(
            m=np.zeros(3),
            n=n,
            inner_radius=float(params.get('radius_top', 1.0)),
            outer_radius=float(params.get('radius_bottom', 5.0)),
            height=float(params.get('height', 10.0)),
            plot=False
        )
        
    elif tool_type == 'Grid':
        gsl = int(params.get('grid_side_length', 10))
        layers = create_Grid(m=np.zeros(3), grid_size_list=[gsl], plot=False)
        return layers[0] if layers else np.array([])
    
    return np.array([])





class Node:
    def __init__(self, function_generator=None, parameters=None, other=None):
        self.results = {}
        self.parent = other if other else None
        self.prev = []
        
        default_params = {
            "polynom_max_power": 5, "neuron_models":["iaf_psc_alpha"], "encoded_polynoms": [],
            "m": [0.0, 0.0, 0.0], "id": -1, "graph_id":0, "name": "Node", "types": [0],
            "auto_spike_recorder": False, "auto_multimeter": False,
            "devices": [],
            "displacement": [0.0, 0.0, 0.0], 
            "displacement_factor": 1.0
        }
        self.parameters = parameters if parameters else default_params.copy()
        
        if 'auto_spike_recorder' not in self.parameters: self.parameters['auto_spike_recorder'] = False
        if 'auto_multimeter' not in self.parameters: self.parameters['auto_multimeter'] = False
        
        self.connections = self.parameters.get('connections', [])
        self.devices = self.parameters.get("devices", [])
        
        self.center_of_mass = np.array(self.parameters.get("m", [0,0,0]), dtype=float)
        self.old_center_of_mass = self.center_of_mass.copy()
        
        self.id = self.parameters.get("id", -1)
        self.name = self.parameters.get("name", f"Node:{self.id}")
        self.types = self.parameters.get("types", [])
        self.graph_id = self.parameters.get("graph_id", 0)
        self.neuron_models = self.parameters.get("neuron_models", ["iaf_psc_alpha"])
        self.population_nest_params = self.parameters.get("population_nest_params", [])

        self.next = []
        if other:
            self.prev.append(other)
            other.next.append(self)
            
        self.function_generator = None
        self.check_function_generator(function_generator, self.parameters.get("polynom_max_power", 5))
        self.positions = []
        self.population = []
        self.nest_connections = []
        self.nest_references = {}

    def set_graph_id(self, graph_id=0): self.parameters["graph_id"] = graph_id; self.graph_id = graph_id
    def check_function_generator(self, func_gen=None, n=5):
        if func_gen: self.function_generator = func_gen; self.coefficients = func_gen.coefficients.copy()
        else: self.function_generator = PolynomGenerator(n=n); self.coefficients = self.function_generator.coefficients.copy()

    def build(self):

        params = self.parameters
        tool_type = params.get('tool_type', 'custom')
        
        local_center_pos = np.array(params.get('m', [0.0, 0.0, 0.0]), dtype=float)
        
        raw_disp = np.array(params.get('displacement', [0.0, 0.0, 0.0]), dtype=float)
        disp_factor = float(params.get('displacement_factor', 1.0))
        final_displacement_vector = raw_disp * disp_factor
        
        print(f"   [Build] Node {self.id}: Local Flow Origin = {local_center_pos}, Global Shift = {final_displacement_vector}")
        
        rot_theta = float(params.get('rot_theta', 0.0))
        rot_phi = float(params.get('rot_phi', 0.0))
        
        sx = float(params.get('stretch_x', 1.0))
        sy = float(params.get('stretch_y', 1.0))
        sz = float(params.get('stretch_z', 1.0))
        transform_matrix = np.diag([sx, sy, sz])
        self.parameters['transform_matrix'] = transform_matrix.tolist()

        generated_positions = []

        if tool_type == 'custom':
            types = params.get('types', [0])
            num_types = len(types)
            grid_size = tuple(params.get('grid_size', [10, 10, 10]))
            
            encoded_per_type = params.get("encoded_polynoms_per_type", None)
            if encoded_per_type:
                if len(encoded_per_type) != num_types:
                    encoded_per_type = encoded_per_type[:num_types]
                
                flow_functions = []
                for type_polynoms in encoded_per_type:
                    try:
                        decoded = self.function_generator.decode_multiple(type_polynoms)
                        flow_functions.append(tuple(decoded))
                    except Exception as e:
                        print(f"    ⚠ Polynomial decode error: {e}. Using identity.")
                        flow_functions.append((lambda x,y,z: x, lambda x,y,z: y, lambda x,y,z: z))
            else:
                flow_functions = [(lambda x,y,z: x, lambda x,y,z: y, lambda x,y,z: z) for _ in range(num_types)]
            
            wave_params = params.get('wave_params', {
                'sparse_holes': params.get('sparse_holes', 0),
                'sparsity_factor': params.get('sparsity_factor', 0.9),
                'probability_vector': params.get('probability_vector', [1.0]*num_types)
            })

            try:
                generated_positions = cluster_and_flow(
                    grid_size=grid_size,
                    m=local_center_pos,
                    rot_theta=rot_theta,
                    rot_phi=rot_phi,
                    transform_matrix=transform_matrix,
                    wave_params=wave_params,
                    types=types,
                    flow_functions=flow_functions,
                    dt=params.get('dt', 0.01),
                    old=params.get('old', True),
                    num_steps=params.get('num_steps', 8),
                    plot_clusters=False,
                    title=params.get('title', "node")
                )
            except Exception as e:
                print(f"    ⚠ WFC Generation Failed: {e}. Creating fallback blob.")
                generated_positions = [blob_positions(n=100, m=local_center_pos, r=2.0, plot=False) for _ in types]

        else: 
            types = params.get('types', [0])
            generated_positions = []
            for i in range(len(types)):
                raw_points = get_raw_shape_points(tool_type, params)
                if len(raw_points) == 0:
                    generated_positions.append(np.array([]))
                    continue
                final_points = transform_points(
                    raw_points, m=local_center_pos,
                    rot_theta=rot_theta, rot_phi=rot_phi,
                    transform_matrix=transform_matrix, plot=False
                )
                generated_positions.append(final_points)


        all_points_flat = [p for p in generated_positions if p is not None and len(p) > 0]
        
        if all_points_flat:
            combined = np.vstack(all_points_flat)
            current_centroid = np.mean(combined, axis=0)
            
            drift_correction = local_center_pos - current_centroid

            total_shift = drift_correction + final_displacement_vector
            
            self.positions = []
            for cluster in generated_positions:
                if cluster is not None and len(cluster) > 0:
                    self.positions.append(cluster + total_shift)
                else:
                    self.positions.append(cluster)
                    

            self.center_of_mass = local_center_pos + final_displacement_vector
            
        else:
            self.positions = generated_positions
            self.center_of_mass = local_center_pos + final_displacement_vector


        self.parameters['m'] = local_center_pos.tolist() 
        self.parameters['center_of_mass'] = self.center_of_mass.tolist()
        self.parameters['old_center_of_mass'] = self.center_of_mass.tolist()

    def populate_node(self):
        print(f"\nPopulating Node {self.id} ({self.name})...")
        params = self.parameters
        tool_type = params.get('tool_type', 'custom')
        
        # FIX: Also check node attribute as fallback
        pop_nest_params = params.get("population_nest_params", [])
        if not pop_nest_params and hasattr(self, 'population_nest_params'):
            pop_nest_params = self.population_nest_params
        
        # Debug: Show what parameters we're using
        if pop_nest_params:
            print(f"  Using custom NEST params for {len(pop_nest_params)} population(s):")
            for idx, p in enumerate(pop_nest_params):
                if p:
                    key_params = {k: p[k] for k in ['V_th', 'C_m', 't_ref', 'E_L'] if k in p}
                    print(f"    Pop {idx}: {key_params}")
        else:
            print(f"  Using NEST defaults (no custom params)")
        
        neuron_params = pop_nest_params[0] if pop_nest_params else {}
        current_models = params.get("neuron_models", ["iaf_psc_alpha"])
        
        created_pops = []
        
        k_val = float(params.get('k', 10.0))
        use_index = params.get('old', False)

        center_pos = self.center_of_mass 
        
        try:
            if tool_type == 'CCW':
                for i, _ in enumerate(self.positions):
                    mod = current_models[i] if i < len(current_models) else "iaf_psc_alpha"

                    
                    pop = create_CCW(
                        model=mod, neuron_params=neuron_params,
                        weight_ex=float(params.get('ccw_weight_ex', 30.0)),
                        delay_ex=float(params.get('ccw_delay_ex', 1.0)),
                        k=k_val,
                        n_neurons=int(params.get('n_neurons', 100)),
                        radius=float(params.get('radius', 5.0)),
                        center=center_pos, # Nutzt das verschobene Zentrum
                        rot_theta=float(params.get('rot_theta', 0.0)),
                        rot_phi=float(params.get('rot_phi', 0.0)),
                        bidirectional=bool(params.get('bidirectional', False)),
                        use_index_based=use_index
                    )
                    created_pops.append(pop)
            elif tool_type == 'Cone':
                for i, _ in enumerate(self.positions):
                    mod = current_models[i] if i < len(current_models) else "iaf_psc_alpha"
                    pop = connect_cone(
                        model=mod, neuron_params=neuron_params,
                        weight_ex=float(params.get('ccw_weight_ex', 30.0)),
                        k=k_val,
                        n_neurons=int(params.get('n_neurons', 100)),
                        radius_top=float(params.get('radius_top', 1.0)),
                        radius_bottom=float(params.get('radius_bottom', 5.0)),
                        height=float(params.get('height', 10.0)),
                        center=center_pos, # Nutzt das verschobene Zentrum
                        rot_theta=float(params.get('rot_theta', 0.0)),
                        rot_phi=float(params.get('rot_phi', 0.0)),
                        bidirectional=bool(params.get('bidirectional', False)),
                        use_index_based=use_index
                    )
                    created_pops.append(pop)
            elif tool_type == 'Blob':
                # Blob nutzt direkt die Positionen aus self.positions (bereits verschoben)
                # Aber create_blob_population erwartet positions array.
                # Wir müssen sicherstellen, dass wir nicht doppelt generieren.
                # create_blob_population nimmt positions argument.
                for i, pos_cluster in enumerate(self.positions):
                    mod = current_models[i] if i < len(current_models) else "iaf_psc_alpha"
                    pop = create_blob_population(pos_cluster, neuron_type=mod, neuron_params=neuron_params)
                    created_pops.append(pop)
            else:
                # Custom / WFC / Grid
                if len(current_models) < len(self.positions):
                    last = current_models[-1] if current_models else "iaf_psc_alpha"
                    current_models.extend([last] * (len(self.positions) - len(current_models)))
                
                # Hier werden die Positionen aus self.positions verwendet, die in build() 
                # bereits durch Flow + Displacement gelaufen sind.
                created_pops = clusters_to_neurons(self.positions, current_models, pop_nest_params, set_positions=True)

            self.population = created_pops

            auto_spikes = params.get('auto_spike_recorder', False)
            auto_volt = params.get('auto_multimeter', False)
            
            if auto_spikes or auto_volt:
                if 'devices' not in self.parameters: self.parameters['devices'] = []
                
                for pop_idx, pop in enumerate(self.population):
                    if pop is None: continue
                    
                    model = current_models[pop_idx] if pop_idx < len(current_models) else "unknown"
                    is_spiking = model not in ['siegert_neuron', 'mcculloch_pitts_neuron', 'rate_neuron_ipn', 'rate_neuron_opn']
                    
                    if auto_spikes and is_spiking:
                        has_rec = any(d.get('model') == 'spike_recorder' and d.get('target_pop_id') == pop_idx for d in self.parameters['devices'])
                        if not has_rec:
                            print(f"  + Auto-Adding Spike Recorder to Pop {pop_idx}")
                            self.parameters['devices'].append({
                                "id": len(self.parameters['devices']),
                                "model": "spike_recorder",
                                "target_pop_id": pop_idx,
                                "params": {"label": f"auto_spikes_p{pop_idx}"},
                                "conn_params": {"weight": 1.0, "delay": 1.0},
                                "runtime_gid": None
                            })
                    
                    if auto_volt:
                        has_multi = any(d.get('model') == 'multimeter' and d.get('target_pop_id') == pop_idx for d in self.parameters['devices'])
                        if not has_multi:
                            print(f"  + Auto-Adding Multimeter to Pop {pop_idx}")
                            recordables = get_mc_recordables(model)
                            self.parameters['devices'].append({
                                "id": len(self.parameters['devices']),
                                "model": "multimeter",
                                "target_pop_id": pop_idx,
                                "params": {"interval": 1.0, "record_from": recordables},
                                "conn_params": {"weight": 1.0, "delay": 1.0},
                                "runtime_gid": None
                            })

            self.instantiate_devices()
            self.verify_and_report()

        except Exception as e:
            print(f" CRITICAL ERROR in populate_node: {e}")
            import traceback; traceback.print_exc()

    def add_neighbor(self, other):
        if other and other not in self.next: self.next.append(other)
        if other and self not in other.prev: other.prev.append(self)

    def remove_neighbor_if_isolated(self, other):
        if not other: return
        connected = False
        for c in self.connections:
            t = c.get('target',{})
            if t.get('graph_id')==other.graph_id and t.get('node_id')==other.id: connected=True; break
        if not connected:
            if other in self.next: self.next.remove(other)
            if self in other.prev: other.prev.remove(self)

    def remove(self):
        for p in self.prev:
            if self in p.next: p.next.remove(self)
        for n in self.next:
            if self in n.prev: n.prev.remove(self)
        self.prev=[]; self.next=[]
        
    def instantiate_devices(self):
        if not hasattr(self, 'devices') or not self.devices:
            if 'devices' in self.parameters: self.devices = self.parameters['devices']
            else: return
            
        print(f" Instantiating {len(self.devices)} devices for Node {self.id}")
        for dev_conf in self.devices:
            try:
                model = dev_conf['model']
                d_params = dev_conf.get('params', {}).copy()  # Copy to avoid modifying original
                
                idx = dev_conf.get('target_pop_id', 0)
                target_model = None
                
                # Get target model for MC detection
                if idx < len(self.population) and self.population[idx]:
                    try:
                        target_model = nest.GetStatus(self.population[idx][0], 'model')[0]
                    except:
                        pass
                
                # Fix recordables for multimeters targeting MC models
                if 'multimeter' in model and target_model in MC_MODELS:
                    record_from = d_params.get('record_from', ['V_m'])
                    if isinstance(record_from, str):
                        record_from = [record_from]
                    # Replace V_m with V_m_s for MC models
                    fixed_recordables = []
                    for r in record_from:
                        if r == 'V_m':
                            fixed_recordables.append('V_m_s')
                        else:
                            fixed_recordables.append(r)
                    d_params['record_from'] = fixed_recordables
                
                nest_dev = nest.Create(model, params=d_params)
                dev_conf['runtime_gid'] = nest_dev
                
                if idx < len(self.population) and self.population[idx]:
                    target = self.population[idx]
                    cp = dev_conf.get('conn_params', {})
                    w = float(cp.get('weight', 1.0))
                    d = max(float(cp.get('delay', 1.0)), 0.1)
                    syn = {'weight': w, 'delay': d}
                    
                    # Auto-detect MC models for generators
                    if "generator" in model and target_model in MC_MODELS:
                        receptor = get_receptor_type_for_model(target_model, excitatory=(w >= 0))
                        if receptor > 0:
                            syn['receptor_type'] = receptor
                    
                    if "generator" in model or "meter" in model: 
                        nest.Connect(nest_dev, target, syn_spec=syn)
                    else: 
                        nest.Connect(target, nest_dev, syn_spec=syn)

            except Exception as e: print(f"Error creating device {dev_conf.get('model')}: {e}")

    def verify_and_report(self, verbose=False):
        if not self.population: return
        for i, pop in enumerate(self.population):
            if pop is None: print(f"Node {self.id} Pop {i}: Creation failed")
            elif verbose: print(f"Node {self.id} Pop {i}: OK")

    def connect(self, graph_registry=None):

        if not self.connections:
            return
        
        print(f"  Connecting Node {self.id} ({len(self.connections)} connections)...")
        
        for conn in self.connections:
            try:
                self._build_single_connection(conn, graph_registry)
            except Exception as e:
                print(f"    ⚠ Connection {conn.get('id', '?')}: {e}")

    def _build_single_connection(self, conn: dict, graph_registry=None):
        src_info = conn.get('source', {})
        tgt_info = conn.get('target', {})
        params = conn.get('params', {})
        
        src_graph_id = src_info.get('graph_id', self.graph_id)
        src_node_id = src_info.get('node_id', self.id)
        src_pop_id = src_info.get('pop_id', 0)
        
        tgt_graph_id = tgt_info.get('graph_id', self.graph_id)
        tgt_node_id = tgt_info.get('node_id', self.id)
        tgt_pop_id = tgt_info.get('pop_id', 0)
        
        if src_graph_id == self.graph_id and src_node_id == self.id:
            if src_pop_id >= len(self.population) or self.population[src_pop_id] is None:
                return
            src_pop = self.population[src_pop_id]
        elif graph_registry:
            src_graph = graph_registry.get(src_graph_id)
            if not src_graph:
                return
            src_node = src_graph.get_node(src_node_id)
            if not src_node or src_pop_id >= len(src_node.population):
                return
            src_pop = src_node.population[src_pop_id]
        else:
            return
        
        if tgt_graph_id == self.graph_id and tgt_node_id == self.id:
            if tgt_pop_id >= len(self.population) or self.population[tgt_pop_id] is None:
                return
            tgt_pop = self.population[tgt_pop_id]
        elif graph_registry:
            tgt_graph = graph_registry.get(tgt_graph_id)
            if not tgt_graph:
                return
            tgt_node = tgt_graph.get_node(tgt_node_id)
            if not tgt_node or tgt_pop_id >= len(tgt_node.population):
                return
            tgt_pop = tgt_node.population[tgt_pop_id]
        else:
            return
        
        if src_pop is None or tgt_pop is None:
            return
        
        self._nest_connect(src_pop, tgt_pop, params)

    def _nest_connect(self, src_pop, tgt_pop, params: dict):
        rule = params.get('rule', 'all_to_all')
        syn_model = params.get('synapse_model', 'static_synapse')
        weight = params.get('weight', 1.0)
        delay = max(params.get('delay', 1.0), 0.1)
        use_spatial = params.get('use_spatial', False)
        
        syn_spec = {
            'synapse_model': syn_model,
            'weight': weight,
            'delay': delay
        }
        
        if syn_model == 'stdp_synapse':
            if 'tau_plus' in params: syn_spec['tau_plus'] = params['tau_plus']
            if 'lambda' in params: syn_spec['lambda'] = params['lambda']  
            if 'alpha' in params: syn_spec['alpha'] = params['alpha']
            if 'mu_plus' in params: syn_spec['mu_plus'] = params['mu_plus']
            if 'mu_minus' in params: syn_spec['mu_minus'] = params['mu_minus']
            if 'Wmax' in params: syn_spec['Wmax'] = params['Wmax']
        
        if syn_model == 'bernoulli_synapse' and 'p_transmit' in params and not use_spatial:
            syn_spec['p_transmit'] = params['p_transmit']
        
        if use_spatial:
            conn_spec = self._build_spatial_conn_spec(params)
        else:
            conn_spec = {'rule': rule}
            if rule == 'fixed_indegree':
                conn_spec['indegree'] = int(params.get('indegree', 10))
            elif rule == 'fixed_outdegree':
                conn_spec['outdegree'] = int(params.get('outdegree', 10))
            elif rule == 'pairwise_bernoulli':
                conn_spec['p'] = float(params.get('p', 0.1))
            conn_spec['allow_autapses'] = params.get('allow_autapses', True)
            conn_spec['allow_multapses'] = params.get('allow_multapses', True)
        
        nest.Connect(src_pop, tgt_pop, conn_spec, syn_spec)

    def _build_spatial_conn_spec(self, params: dict) -> dict:
        """Baut Connection Spec mit räumlicher Maske (NEST 3.x kompatibel)."""
        mask_type = params.get('mask_type', 'sphere')
        mask_radius = float(params.get('mask_radius', 1.0))
        inner_radius = float(params.get('mask_inner_radius', 0.0))
        
        if mask_type in ('sphere', 'spherical'):
            mask = nest.CreateMask('spherical', {'radius': mask_radius})
            if inner_radius > 0:
                inner_mask = nest.CreateMask('spherical', {'radius': inner_radius})
                mask = mask & ~inner_mask  
        elif mask_type in ('box', 'rectangular'):
            half = mask_radius
            mask = nest.CreateMask('rectangular', {
                'lower_left': [-half, -half, -half],
                'upper_right': [half, half, half]
            })
        else:
            mask = nest.CreateMask('spherical', {'radius': mask_radius})
        
        return {
            'rule': 'pairwise_bernoulli',
            'p': float(params.get('p', 1.0)),
            'mask': mask,
            'allow_autapses': params.get('allow_autapses', True),
            'allow_multapses': params.get('allow_multapses', True)
        }









def generate_node_parameters_list(n_nodes=5,
                                   n_types=5,
                                   vary_polynoms=True,
                                   vary_types_per_node=True,
                                   safe_mode=True,
                                   max_power=2,
                                   max_coeff=0.8,
                                   graph_id=0,
                                   add_self_connections=False,
                                   self_conn_probability=0.3):
    
    params_list = []
    
    for i in range(n_nodes):
        if vary_types_per_node:
            node_n_types = np.random.randint(1, n_types + 1)
        else:
            node_n_types = n_types
        
        types = list(range(node_n_types))
        
        available_models = ["iaf_psc_alpha", "iaf_psc_exp", "iaf_psc_delta"]
        neuron_models = [available_models[i % len(available_models)]
                        for i in range(node_n_types)]
        
        probability_vector = list(np.random.dirichlet([1] * node_n_types))
        
        params = {
            "types": types,
            "neuron_models": neuron_models,
            "grid_size": [
                np.random.randint(8, 15),
                np.random.randint(8, 15),
                np.random.randint(8, 15)
            ],
            "m": [
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-1.0, 1.0)
            ],
            "rot_theta": np.random.uniform(-np.pi, np.pi),
            "rot_phi": np.random.uniform(-np.pi, np.pi),
            "transform_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "dt": np.random.uniform(0.005, 0.02),
            "old": True,
            "num_steps": np.random.randint(5, 12),
            "plot_clusters": False,
            "title": f"Node_{i}",
            "sparse_holes": np.random.randint(0, 3),
            "sparsity_factor": np.random.uniform(0.85, 0.95),
            "probability_vector": probability_vector,
            "name": f"Node_{i}",
            "id": i,
            "graph_id": graph_id,
            "distribution": [],
            "conn_prob": [],
            "polynom_max_power": 5,
            "coefficients": None,
            "center_of_mass": np.array([0.0, 0.0, 0.0]),
            "displacement": np.array([0.0, 0.0, 0.0]),
            "displacement_factor": 1.0,
            "field": None,
            "connections": []
        }

        if add_self_connections and np.random.random() < self_conn_probability:
            for pop_id in range(node_n_types):
                self_conn = {
                    'id': len(params['connections']) + 1,
                    'name': f'SelfConn_N{i}_P{pop_id}',
                    'source': {
                        'graph_id': graph_id,
                        'node_id': i,
                        'pop_id': pop_id
                    },
                    'target': {
                        'graph_id': graph_id,
                        'node_id': i,
                        'pop_id': pop_id
                    },
                    'params': {
                        'rule': 'fixed_indegree',
                        'indegree': np.random.randint(1, 5),
                        'synapse_model': 'static_synapse',
                        'weight': np.random.uniform(0.5, 2.0),
                        'delay': 1.0,
                        'allow_autapses': False,
                        'allow_multapses': True
                    }
                }
                params['connections'].append(self_conn)
        
        if vary_polynoms:
            encoded_polynoms = []
            for type_idx in range(node_n_types):
                type_polynoms = []
                for coord in range(3):
                    num_terms = np.random.randint(2, 5)
                    indices = []
                    coeffs = []
                    for _ in range(num_terms):
                        idx = np.random.choice([0, 1, 2])
                        if safe_mode:
                            power = np.random.choice(range(min(max_power + 1, 4)))
                            coeff = np.random.uniform(-max_coeff, max_coeff)
                        else:
                            power = np.random.choice([0, 1, 2, 3])
                            coeff = np.random.randn() * 0.5
                        indices.append([idx, power])
                        coeffs.append(float(coeff))
                    
                    poly_encoded = {
                        'indices': indices,
                        'coefficients': coeffs,
                        'n': 5,
                        'decay': 0.5
                    }
                    type_polynoms.append(poly_encoded)
                encoded_polynoms.append(type_polynoms)
            params["encoded_polynoms_per_type"] = encoded_polynoms
        else:
            identity_polynoms = []
            for type_idx in range(node_n_types):
                type_polynoms = [
                    {'indices': [[0, 1]], 'coefficients': [1.0], 'n': 5, 'decay': 0.5},
                    {'indices': [[1, 1]], 'coefficients': [1.0], 'n': 5, 'decay': 0.5},
                    {'indices': [[2, 1]], 'coefficients': [1.0], 'n': 5, 'decay': 0.5}
                ]
                identity_polynoms.append(type_polynoms)
            params["encoded_polynoms_per_type"] = identity_polynoms
        
        params_list.append(params)
    
    return params_list

class Graph:
    def __init__(self,
                 graph_name="",
                 graph_id=0,
                 parameter_list=[],
                 polynom_max_power=5,
                 polynom_decay=0.8,
                 position=None,
                max_nodes=None):

        self.node_list = []
        self.node_dict = {}
        self.graph_id=graph_id
        self.nodes = 0
        self.graph_name=graph_name
        self._next_id = 0
        self.max_nodes=max_nodes if max_nodes else len(parameter_list)
        
        if position is not None:
            self.init_position = np.array(position)
        elif parameter_list and len(parameter_list) > 0:
            self.init_position = np.array(parameter_list[0].get("m", [0.0, 0.0, 0.0]))
        else:
            self.init_position = np.array([0.0, 0.0, 0.0])
        
        self.parameter_list = parameter_list if parameter_list else []
        self.polynom_max_power = polynom_max_power
        self.polynom_decay = polynom_decay
        self.polynom_generator = PolynomGenerator(n=polynom_max_power, decay=polynom_decay)
        
        self.root = None
    
    def random_direction(self, length=1.0):
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)
        return v * length
    
    def create_node(self,
                    parameters=None,
                    other=None,
                    is_root=False,
                    displacement=None,
                    displacement_factor=1.0,
                    auto_build=False):
        node_id = self._next_id
        
        if parameters:
            params = parameters.copy()
        else:
            params = {}
        
        actual_factor = params.get('displacement_factor', displacement_factor)

        params['id'] = node_id
        if 'name' not in params or params.get('name', '').startswith('Node_'):
            params['name'] = f"Node_{node_id}"
        
        if other is not None:
            has_explicit_position = (
                parameters and 'm' in parameters and
                not np.allclose(parameters['m'], [0.0, 0.0, 0.0])
            )
            
            if not has_explicit_position:
                if displacement is None:
                    displacement = self.random_direction(actual_factor)
                
                new_position = np.array(other.center_of_mass) + displacement
                params['m'] = new_position.tolist()
                params['center_of_mass'] = new_position.tolist()
        
        elif not is_root and 'm' not in params:
            params['m'] = [0.0, 0.0, 0.0]
            params['center_of_mass'] = [0.0, 0.0, 0.0]
        
        if is_root or other is None:
            function_generator = self.polynom_generator
        else:
            function_generator = None
        
        new_node = Node(
            function_generator=function_generator,
            parameters=params,
            other=other
        )
        
        new_node.id = node_id
        new_node.name = params['name']
        
        self.node_dict[new_node.name] = new_node
        self.node_dict[new_node.id] = new_node
        
        self._next_id += 1
        self.node_list.append(new_node)
        self.nodes += 1
        if self.root is None:
            self.root = new_node
        
        if auto_build:
            try:
                new_node.build()
            except Exception as e:
                print(f"⚠ Build failed for Node {node_id}: {e}")
        
        return new_node
    

    def populate(self,
                 n_iterations=5,
                 displacement_function=None,
                 displacement_factor=1.0,
                 parameters_dict=None,
                 parameters_list=None,
                 node_names=None,
                 random_parent=True,
                 mutation=False,
                 auto_build=False):
        
        if displacement_function is None:
            displacement_function = self.random_direction
        
        if parameters_dict is None:
            if parameters_list is not None:
                parameters_dict = {i: params for i, params in enumerate(parameters_list)}
            elif self.parameter_list:
                parameters_dict = {i: params for i, params in enumerate(self.parameter_list)}
            else:
                raise ValueError("No parameters available")
        
        if node_names is None:
            node_names = {}
        
        root_params = parameters_dict.get(0, list(parameters_dict.values())[0])
        root_params['name'] = node_names.get(0, "Root")
        root_params['m'] = self.init_position.tolist()
        
        self.root = self.create_node(
            parameters=root_params,
            is_root=True,
            auto_build=auto_build
        )
        
        for iteration in range(n_iterations):
            current_nodes = self.node_list.copy()
            n_current = len(current_nodes)
            
            if random_parent:
                p = rand_prob_vector(n_current)
            else:
                p = None
            
            created_count = 0
            
            for node in current_nodes:
                if self.nodes >= self.max_nodes:
                    break
                
                next_id = self._next_id
                
                if next_id not in parameters_dict:
                    if mutation and node.parent is not None:
                        node_params = node.parent.parameters.copy()
                    else:
                        continue
                else:
                    node_params = parameters_dict[next_id]
                
                parent = np.random.choice(current_nodes, p=p) if random_parent else node
                
                node_params['name'] = node_names.get(next_id, f"Node_{next_id}")
                
                new_node = self.create_node(
                    parameters=node_params,
                    other=parent,
                    displacement=displacement_function(displacement_factor),
                    auto_build=auto_build
                )
                
                if mutation and new_node.parent is not None:
                    new_node.coefficients = new_node.update_coefficients(
                        coefficients=new_node.parent.coefficients
                    )
                    new_node.function_generator.coefficients = new_node.coefficients
                
                created_count += 1
            
            if created_count == 0 or self.nodes >= self.max_nodes:
                break


    def build_connections(self, external_graphs=None):

        graph_registry = {self.graph_id: self}
        if external_graphs:
            graph_registry.update(external_graphs)
        
        for node in self.node_list:
            node.connect(graph_registry)
                
    def build_all(self):
        for i, node in enumerate(self.node_list, 1):
            try:
                node.build()
                node.already_built = False
            except Exception as e:
                print(f"✗ Error: {e}")
    
    def get_node(self, identifier):
        if isinstance(identifier, int):
            identifier = identifier
        return self.node_dict.get(identifier)
    
    def remove_node(self, node):
        if node in self.node_list:
            node.remove()
            self.node_list.remove(node)
            
            if node.name in self.node_dict:
                del self.node_dict[node.name]
            if str(node.id) in self.node_dict:
                del self.node_dict[str(node.id)]
            
            self.nodes -= 1
            print(f"Node {node.id} ({node.name}) deleted")
        else:
            print(f"No node with id {node.id} found")
    
    def connect_nodes(self, from_node, to_node):
        from_node.add_edge_to(to_node)
    
    def list_nodes(self):
        print(f"{'ID':<4} {'Name':<15} {'Position':<25} {'Parent':<10} {'Children':<10}")
        print("-" * 75)
        for node in self.node_list:
            pos_str = f"({node.center_of_mass[0]:.2f}, {node.center_of_mass[1]:.2f}, {node.center_of_mass[2]:.2f})"
            parent_str = f"{node.parent.id}" if node.parent else "None"
            children_str = f"{len(node.next)}"
            pop_str = f", Pop: {len(node.population)}" if node.population else ""
            print(f"{node.id:<4} {node.name:<15} {pos_str:<25} {parent_str:<10} {children_str:<10}{pop_str}")
    

    

    def __repr__(self):
        return f"Graph(nodes={self.nodes}, edges={sum(len(n.next) for n in self.node_list)})"


def createGraph(parameter_list=None, max_nodes=10, graph_id=0):
    parameter_list = parameter_list if parameter_list is not None else generate_node_parameters_list(n_nodes=max_nodes, n_types=3, graph_id=graph_id)
    max_nodes = len(parameter_list)
    graph = Graph(parameter_list=parameter_list, graph_id=graph_id)
    graph.populate()
    return graph

def graphInfo(graph, figsize_per_plot=(5, 4), marker_size=10, alpha=0.6, linewidths=0.5):
    n_nodes = len(graph.node_list)
    if n_nodes == 0:
        return
    plot_graph_3d(graph)
    n_cols = int(np.ceil(np.sqrt(n_nodes)))
    n_rows = int(np.ceil(n_nodes / n_cols))
    
    fig = plt.figure(figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    
    for idx, node in enumerate(graph.node_list):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        clusters = node.positions
        n_types = len(clusters)
        cmap_obj = plt.get_cmap('tab10')
        colors = [cmap_obj(i / max(n_types - 1, 1)) for i in range(n_types)]
        
        for pts, col in zip(clusters, colors):
            pts = np.asarray(pts)
            if pts.size > 0:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                          c=[col], s=marker_size,
                          edgecolor='k', alpha=alpha, linewidths=linewidths)
        
        type_info = f"Types: {node.types}" if len(node.types) <= 5 else f"Types: {len(node.types)}"
        ax.set_title(f"{node.name}\n{type_info}", fontsize=10)
        
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.tick_params(labelsize=7)
    
    for idx in range(n_nodes, n_rows * n_cols):
        fig.add_subplot(n_rows, n_cols, idx + 1).axis('off')
    
    plt.tight_layout()
    plt.show()
    pos = []


    for node in graph.node_list:
        for x in node.positions:
            pos.append(x)
    plot_point_clusters(pos)


def get_ccw_topology(node_id, pop_id, num_neurons, weight=10.0, delay=1.0):

    connections = []
    
    conn = {
        'id': 0,
        'name': f'CCW_Ring_{node_id}',
        'source': {'graph_id': 0, 'node_id': node_id, 'pop_id': pop_id},
        'target': {'graph_id': 0, 'node_id': node_id, 'pop_id': pop_id},
        'params': {
            'rule': 'fixed_outdegree',
            'outdegree': 1,
            'synapse_model': 'static_synapse',
            'weight': weight,
            'delay': delay,
            'topology_type': 'ring_ccw'
        }
    }
    return [conn]

def get_shape_positions(tool_type, params):

    center = np.array(params.get('m', [0,0,0]))
    
    if tool_type == 'CCW':
        n = params.get('n_neurons', 100)
        r = params.get('radius', 5.0)
        return circle3d(n=n, r=r, m=center, plot=False)
        
    elif tool_type == 'Blob':
        n = params.get('n_neurons', 100)
        return blob_positions(n=n, m=center, r=params.get('radius', 5.0), plot=False)
        
    elif tool_type == 'Cone':
        return create_cone(
            m=center,
            n=params.get('n_neurons', 100),
            inner_radius=params.get('radius_top', 1.0),
            outer_radius=params.get('radius_bottom', 5.0),
            height=params.get('height', 10.0),
            plot=False
        )
        
    elif tool_type == 'Grid':
        gsl = params.get('grid_side_length', 10)
        layers = create_Grid(m=center, grid_size_list=[gsl], plot=False)
        return layers[0]
        
    return np.array([])

def get_connection_snapshot(population):
    conns = nest.GetConnections(population, population)
    if not conns:
        return set()
    
    sources = conns.source
    targets = conns.target
    return set(zip(sources, targets))

def sync_node_attributes_to_parameters(node):

    if hasattr(node, 'types'):
        node.parameters['types'] = node.types
    if hasattr(node, 'neuron_models'):
        node.parameters['neuron_models'] = node.neuron_models
    if hasattr(node, 'center_of_mass'):
        node.parameters['center_of_mass'] = list(node.center_of_mass)
        node.parameters['m'] = list(node.center_of_mass)
