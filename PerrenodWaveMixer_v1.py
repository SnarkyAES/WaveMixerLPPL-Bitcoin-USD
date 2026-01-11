# ============================================================
# PERRENOD WAVE MIXER v1.0
# ============================================================
# Interactive wave mixer for Bitcoin price modeling
# Based on the work of Giovanni Santostasi and Stephen Perrenod
# 
# Formula: ln(P) = α×ln(t) + c + Σ Aᵢ×t^βᵢ × cos(hᵢ×ω×ln(t) + φᵢ)
#
# Author: Snarky (snarkyaes@proton.me)
# ============================================================

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import os
import requests
from datetime import datetime
from scipy.optimize import differential_evolution
import threading

# ============================================================
# CONSTANTS
# ============================================================

GENESIS_DATE = pd.Timestamp("2009-01-03")
LAMBDA = 2.0
OMEGA = 2 * np.pi / np.log(LAMBDA)  # ≈ 9.0647

# Available harmonics for selection
AVAILABLE_HARMONICS = [
    (0.25, "1/4 sub (~16y)"),
    (0.333, "1/3 sub (~12y)"),
    (0.5, "1/2 sub (~8y)"),
    (0.667, "2/3 sub (~6y)"),
    (1, "FUND (~4y)"),
    (1.5, "3/2 harm (~2.7y)"),
    (2, "2nd harm (~2y)"),
    (3, "3rd harm (~1.3y)"),
    (4, "4th harm (~1y)"),
    (5, "5th harm (~0.8y)"),
    (6, "6th harm (~0.7y)"),
]

# Parameter bounds - designed so defaults are centered
BOUNDS = {
    'alpha': (5.0, 6.4),      # center: 5.7
    'c': (-44, -32),          # center: -38
    'A': (0, 200),            # center: 100
    'beta': (-0.90, -0.10),   # center: -0.50
    'phi': (0, 2 * np.pi),    # center: π
}

# Default parameters (centered in bounds where possible)
DEFAULT_PARAMS = {
    'alpha': 5.70,            # centered
    'c': -38.0,               # centered
    'channels': [
        {'harm_idx': 1,   'A': 100.0, 'beta': -0.50, 'phi': np.pi, 'enabled': True},
        {'harm_idx': 2,   'A': 100.0, 'beta': -0.50, 'phi': np.pi, 'enabled': True},
        {'harm_idx': 3,   'A': 100.0, 'beta': -0.50, 'phi': np.pi, 'enabled': False},
        {'harm_idx': 4,   'A': 100.0, 'beta': -0.50, 'phi': np.pi, 'enabled': False},
    ]
}

# Optimized parameters (for reset to optimized)
OPTIMIZED_PARAMS = {
    'alpha': 5.70,
    'c': -38.04,
    'channels': [
        {'harm_idx': 1,   'A': 135.13, 'beta': -0.69, 'phi': 1.04, 'enabled': True},
        {'harm_idx': 2,   'A': 1.09,   'beta': -0.20, 'phi': 3.69, 'enabled': True},
        {'harm_idx': 3,   'A': 100.0,  'beta': -0.50, 'phi': np.pi, 'enabled': False},
        {'harm_idx': 4,   'A': 100.0,  'beta': -0.50, 'phi': np.pi, 'enabled': False},
    ]
}

DATA_FILE = "btc_usd_data_mixer.csv"

# ============================================================
# DATA FUNCTIONS
# ============================================================

def fetch_btc_data():
    """Fetch BTC/USD data from CryptoCompare"""
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    all_data = []
    to_ts = None
    
    print("Fetching BTC/USD data...")
    while True:
        params = {'fsym': 'BTC', 'tsym': 'USD', 'limit': 2000}
        if to_ts:
            params['toTs'] = to_ts
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        if data['Response'] != 'Success':
            break
        rows = data['Data']['Data']
        if not rows or rows[0]['time'] == 0:
            break
        all_data = rows + all_data
        to_ts = rows[0]['time'] - 1
        if to_ts < 1279324800:
            break
    
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df = df[['date', 'close']].copy()
    df = df[df['close'] > 0].copy()
    return df

def load_data():
    """Load BTC/USD data"""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, parse_dates=['date'])
    else:
        df = fetch_btc_data()
        df.to_csv(DATA_FILE, index=False)
    
    df['days'] = (df['date'] - GENESIS_DATE).dt.days
    df = df[df['days'] > 0].copy()
    return df

# ============================================================
# MODEL FUNCTION
# ============================================================

def perrenod_model(t, alpha, c, channels):
    """
    Perrenod LPPL model
    
    Args:
        t: days since genesis (array)
        alpha: power law slope
        c: intercept
        channels: list of channel dicts with harm_idx, A, beta, phi, enabled
    
    Returns:
        ln(P) array
    """
    ln_t = np.log(t)
    result = alpha * ln_t + c
    
    for ch in channels:
        if ch.get('enabled', True):
            harm_idx = ch['harm_idx']
            A = ch['A']
            beta = ch['beta']
            phi = ch['phi']
            oscillation = A * np.power(t, beta) * np.cos(harm_idx * OMEGA * ln_t + phi)
            result += oscillation
    
    return result

def calc_r2(y_true, y_pred):
    """Calculate R²"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0

# ============================================================
# MAIN APPLICATION
# ============================================================

class WaveMixer:
    def __init__(self, root):
        self.root = root
        self.root.title("🎛️ Perrenod Wave Mixer - BTC/USD")
        self.root.geometry("1500x900")
        self.root.configure(bg='#1a1a2e')
        
        # Load data
        self.df = load_data()
        self.t_full = self.df['days'].values.astype(np.float64)
        self.y_full = np.log(self.df['close'].values)
        self.today_days = (pd.Timestamp.now() - GENESIS_DATE).days
        self.today_date = pd.Timestamp.now()
        
        # Initialize parameters
        self.params = self._copy_params(DEFAULT_PARAMS)
        
        # UI variables
        self.slider_vars = {}
        self.check_vars = {}
        self.combo_vars = {}
        self.channel_frames = []
        
        # Throttle for updates
        self.update_pending = False
        
        # Build UI
        self.build_ui()
        
        # Initial plot
        self.update_plot()
    
    def _copy_params(self, params):
        """Deep copy parameters"""
        import copy
        return copy.deepcopy(params)
    
    def build_ui(self):
        """Build the user interface"""
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls (mixer style)
        left_frame = ttk.Frame(main_frame, width=520)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(left_frame, text="🎛️ WAVE MIXER", 
                               font=('Arial', 16, 'bold'), fg='#4ECDC4', bg='#1a1a2e')
        title_label.pack(pady=(0, 5))
        
        # Stats display - PROMINENTE
        stats_frame = tk.Frame(left_frame, bg='#2d2d44', padx=15, pady=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(stats_frame, text="📊 LIVE METRICS", font=('Arial', 10, 'bold'),
                 fg='#4ECDC4', bg='#2d2d44').pack()
        
        self.r2_var = tk.StringVar(value="R² = 0.0000")
        self.mae_var = tk.StringVar(value="MAE = 0.0000")
        self.bias_var = tk.StringVar(value="Bias = 0.0000")
        
        metrics_row = tk.Frame(stats_frame, bg='#2d2d44')
        metrics_row.pack(fill=tk.X, pady=5)
        
        tk.Label(metrics_row, textvariable=self.r2_var, font=('Consolas', 12, 'bold'),
                 fg='#4ECDC4', bg='#2d2d44').pack(side=tk.LEFT, padx=10)
        tk.Label(metrics_row, textvariable=self.mae_var, font=('Consolas', 12, 'bold'),
                 fg='#45B7D1', bg='#2d2d44').pack(side=tk.LEFT, padx=10)
        tk.Label(metrics_row, textvariable=self.bias_var, font=('Consolas', 12, 'bold'),
                 fg='#96CEB4', bg='#2d2d44').pack(side=tk.LEFT, padx=10)
        
        # Formula display
        formula_frame = tk.Frame(left_frame, bg='#2d2d44', padx=10, pady=5)
        formula_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(formula_frame, text="ln(P) = α·ln(t) + c + Σ Aᵢ·t^βᵢ·cos(hᵢ·ω·ln(t) + φᵢ)", 
                 font=('Consolas', 9), fg='#888', bg='#2d2d44').pack()
        tk.Label(formula_frame, text=f"ω = {OMEGA:.4f} (λ = {LAMBDA})", 
                 font=('Consolas', 9), fg='#666', bg='#2d2d44').pack()
        
        # Scrollable controls area
        canvas = tk.Canvas(left_frame, bg='#1a1a2e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Power Law channel
        self.create_power_law_channel(self.scrollable_frame)
        
        # Harmonic channels
        for i, ch in enumerate(self.params['channels']):
            self.create_harmonic_channel(self.scrollable_frame, i)
        
        # Buttons frame - INSIDE scrollable frame, after channels
        btn_frame = tk.Frame(self.scrollable_frame, bg='#1a1a2e')
        btn_frame.pack(fill=tk.X, pady=15, padx=5)
        
        # First row of buttons
        btn_row1 = tk.Frame(btn_frame, bg='#1a1a2e')
        btn_row1.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_row1, text="🎯 Reset Center", 
                   command=self.reset_center).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_row1, text="⭐ Load Optimized", 
                   command=self.load_optimized).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        
        # Second row of buttons
        btn_row2 = tk.Frame(btn_frame, bg='#1a1a2e')
        btn_row2.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_row2, text="⚡ Optimize from Here", 
                   command=self.optimize_from_current).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_row2, text="📋 Copy Params", 
                   command=self.copy_params).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        
        # Status - also inside scrollable
        self.status_var = tk.StringVar(value="Ready - Move sliders to adjust the model")
        status_label = tk.Label(self.scrollable_frame, textvariable=self.status_var,
                                font=('Arial', 9), fg='#888', bg='#1a1a2e')
        status_label.pack(pady=5)
        
        # Right panel - Graph
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), facecolor='#1a1a2e')
        self.ax = self.fig.add_subplot(111)
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_power_law_channel(self, parent):
        """Create Power Law control channel"""
        frame = tk.LabelFrame(parent, text="⚡ POWER LAW (Base)", 
                              font=('Arial', 10, 'bold'), fg='#FF6B6B', bg='#2d2d44',
                              padx=10, pady=10)
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Alpha slider
        self.create_slider(frame, 'alpha', 'α (slope)', 
                           self.params['alpha'], BOUNDS['alpha'], row=0)
        
        # C slider
        self.create_slider(frame, 'c', 'c (intercept)', 
                           self.params['c'], BOUNDS['c'], row=1)
    
    def create_harmonic_channel(self, parent, channel_idx):
        """Create a harmonic control channel with harmonic selector"""
        ch = self.params['channels'][channel_idx]
        
        # Determine color based on channel
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        color = colors[channel_idx % len(colors)]
        
        frame = tk.LabelFrame(parent, text=f"🎵 CHANNEL {channel_idx + 1}", 
                              font=('Arial', 10, 'bold'), fg=color, bg='#2d2d44',
                              padx=10, pady=10)
        frame.pack(fill=tk.X, pady=5, padx=5)
        self.channel_frames.append(frame)
        
        # Top row: Enable checkbox + Harmonic selector
        top_row = tk.Frame(frame, bg='#2d2d44')
        top_row.grid(row=0, column=0, columnspan=4, sticky=tk.EW, pady=(0, 5))
        
        # Enable checkbox
        var = tk.BooleanVar(value=ch['enabled'])
        self.check_vars[channel_idx] = var
        cb = ttk.Checkbutton(top_row, text="Enable", variable=var,
                             command=self.schedule_update)
        cb.pack(side=tk.LEFT)
        
        # Harmonic selector
        tk.Label(top_row, text="  Harmonic:", font=('Arial', 9), 
                 fg='white', bg='#2d2d44').pack(side=tk.LEFT, padx=(10, 5))
        
        combo_var = tk.StringVar()
        self.combo_vars[channel_idx] = combo_var
        
        # Find current selection
        current_harm = ch['harm_idx']
        current_text = ""
        combo_values = []
        for h_idx, h_name in AVAILABLE_HARMONICS:
            combo_values.append(f"{h_idx}: {h_name}")
            if h_idx == current_harm:
                current_text = f"{h_idx}: {h_name}"
        
        combo = ttk.Combobox(top_row, textvariable=combo_var, values=combo_values,
                             width=18, state='readonly')
        combo.set(current_text)
        combo.pack(side=tk.LEFT)
        combo.bind('<<ComboboxSelected>>', lambda e, idx=channel_idx: self.on_harmonic_change(idx))
        
        # Period info label
        period_days = int(1460 / ch['harm_idx'])
        period_years = period_days / 365.25
        self.period_labels = getattr(self, 'period_labels', {})
        period_var = tk.StringVar(value=f"T≈{period_days}d ({period_years:.1f}y)")
        self.period_labels[channel_idx] = period_var
        tk.Label(top_row, textvariable=period_var, font=('Arial', 8), 
                 fg='#888', bg='#2d2d44').pack(side=tk.RIGHT, padx=5)
        
        # A slider
        self.create_slider(frame, f'A_{channel_idx}', 'A (amplitude)', 
                           ch['A'], BOUNDS['A'], row=1, channel_idx=channel_idx)
        
        # Beta slider
        self.create_slider(frame, f'beta_{channel_idx}', 'β (decay)', 
                           ch['beta'], BOUNDS['beta'], row=2, channel_idx=channel_idx)
        
        # Phi slider
        self.create_slider(frame, f'phi_{channel_idx}', 'φ (phase)', 
                           ch['phi'], BOUNDS['phi'], row=3, channel_idx=channel_idx)
    
    def on_harmonic_change(self, channel_idx):
        """Handle harmonic selection change"""
        combo_var = self.combo_vars[channel_idx]
        selection = combo_var.get()
        
        # Parse "1: FUND (~4y)" -> 1
        harm_idx = float(selection.split(':')[0])
        self.params['channels'][channel_idx]['harm_idx'] = harm_idx
        
        # Update period label
        period_days = int(1460 / harm_idx)
        period_years = period_days / 365.25
        self.period_labels[channel_idx].set(f"T≈{period_days}d ({period_years:.1f}y)")
        
        self.schedule_update()
    
    def create_slider(self, parent, key, label, initial, bounds, row, channel_idx=None):
        """Create a slider control"""
        min_val, max_val = bounds
        
        # Label
        lbl = tk.Label(parent, text=label, font=('Arial', 9), fg='white', 
                       bg='#2d2d44', width=12, anchor='w')
        lbl.grid(row=row, column=0, sticky=tk.W, pady=2)
        
        # Value display
        val_var = tk.StringVar(value=f"{initial:.3f}")
        val_label = tk.Label(parent, textvariable=val_var, font=('Consolas', 9), 
                             fg='#4ECDC4', bg='#2d2d44', width=8)
        val_label.grid(row=row, column=1, padx=5)
        
        # Slider
        slider = ttk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                           length=220, value=initial,
                           command=lambda v, k=key, vv=val_var, ci=channel_idx: 
                                   self.on_slider_change(k, float(v), vv, ci))
        slider.grid(row=row, column=2, columnspan=2, sticky=tk.EW, pady=2)
        
        # Store reference
        self.slider_vars[key] = {'slider': slider, 'val_var': val_var, 'channel_idx': channel_idx}
    
    def on_slider_change(self, key, value, val_var, channel_idx=None):
        """Handle slider change"""
        val_var.set(f"{value:.3f}")
        
        # Update params
        if channel_idx is not None:
            param_name = key.split('_')[0]  # 'A', 'beta', or 'phi'
            self.params['channels'][channel_idx][param_name] = value
        else:
            self.params[key] = value
        
        self.schedule_update()
    
    def schedule_update(self):
        """Schedule a plot update with throttling"""
        if not self.update_pending:
            self.update_pending = True
            self.root.after(50, self.do_update)  # 50ms throttle
    
    def do_update(self):
        """Execute the plot update"""
        self.update_pending = False
        
        # Update enabled states from checkboxes
        for idx, var in self.check_vars.items():
            self.params['channels'][idx]['enabled'] = var.get()
        
        self.update_plot()
    
    def update_plot(self):
        """Update the plot with current parameters"""
        self.ax.clear()
        self.ax.set_facecolor('#1a1a2e')
        
        # Get current parameters
        alpha = self.params['alpha']
        c = self.params['c']
        channels = self.params['channels']
        
        # Calculate model for historical data
        y_fit = perrenod_model(self.t_full, alpha, c, channels)
        
        # Calculate metrics
        r2 = calc_r2(self.y_full, y_fit)
        residuals = self.y_full - y_fit
        mae = np.mean(np.abs(residuals))
        bias = np.mean(residuals)
        
        # Update stats display
        self.r2_var.set(f"R² = {r2:.4f}")
        self.mae_var.set(f"MAE = {mae:.4f}")
        self.bias_var.set(f"Bias = {bias:+.4f}")
        
        # Color code R²
        # (handled by fixed colors for now)
        
        # Projection: 10 years into future
        proj_years = 10
        end_days = self.today_days + proj_years * 365.25
        proj_days = np.linspace(self.t_full.min(), end_days, 1000)
        proj_dates = pd.to_datetime(GENESIS_DATE) + pd.to_timedelta(proj_days, unit='D')
        
        y_proj = perrenod_model(proj_days, alpha, c, channels)
        
        # Plot historical data
        self.ax.semilogy(self.df['date'], self.df['close'], 
                         color='#FFD700', alpha=0.6, lw=0.8, label='BTC/USD')
        
        # Plot model
        self.ax.semilogy(proj_dates, np.exp(y_proj), 
                         color='#4ECDC4', lw=2.5, label='Perrenod Model')
        
        # Plot pure power law for reference
        y_pl = np.exp(alpha * np.log(proj_days) + c)
        self.ax.semilogy(proj_dates, y_pl, 
                         color='#FF6B6B', lw=1.5, ls='--', alpha=0.7, label='Power Law only')
        
        # Today line
        self.ax.axvline(self.today_date, color='yellow', ls=':', alpha=0.8, lw=1)
        
        # Formatting
        self.ax.set_title(f'BTC/USD - Perrenod Wave Model (10y Projection)', 
                          color='white', fontsize=14, fontweight='bold')
        self.ax.set_ylabel('BTC/USD ($)', color='white', fontsize=11)
        self.ax.set_xlabel('Date', color='white', fontsize=11)
        
        # Grid
        self.ax.grid(True, which='major', alpha=0.3)
        self.ax.grid(True, which='minor', alpha=0.1, linestyle=':')
        self.ax.minorticks_on()
        
        # Axis colors
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        
        # Legend
        self.ax.legend(loc='upper left', fontsize=9, facecolor='#2d2d44', 
                       labelcolor='white', framealpha=0.9)
        
        # Price formatter
        def price_formatter(x, pos):
            if x >= 1e6:
                return f'${x/1e6:.1f}M'
            elif x >= 1e3:
                return f'${x/1e3:.0f}k'
            else:
                return f'${x:.0f}'
        
        self.ax.yaxis.set_major_formatter(FuncFormatter(price_formatter))
        
        # Dynamic y-axis limits
        y_max = max(np.exp(y_proj).max(), self.df['close'].max()) * 1.5
        y_min = min(self.df['close'].min(), 0.01)
        self.ax.set_ylim(y_min, y_max)
        
        # Add projection info box
        current_price = self.df['close'].iloc[-1]
        proj_price = np.exp(perrenod_model(np.array([end_days]), alpha, c, channels)[0])
        
        # List enabled harmonics
        enabled_harms = [f"h{ch['harm_idx']}" for ch in channels if ch['enabled']]
        harms_str = ", ".join(enabled_harms) if enabled_harms else "none"
        
        info_text = f"Current: ${current_price:,.0f}\n"
        info_text += f"Proj {proj_years}y: ${proj_price:,.0f}\n"
        info_text += f"α={alpha:.3f}, c={c:.2f}\n"
        info_text += f"Harmonics: {harms_str}"
        
        self.ax.text(0.98, 0.02, info_text, transform=self.ax.transAxes,
                     fontsize=9, color='white', family='monospace',
                     verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='#2d2d44', alpha=0.8))
        
        self.fig.tight_layout()
        self.canvas_plot.draw()
    
    def reset_center(self):
        """Reset all parameters to centered defaults"""
        self.params = self._copy_params(DEFAULT_PARAMS)
        self._update_all_ui()
        self.status_var.set("🎯 Reset to centered defaults")
    
    def load_optimized(self):
        """Load optimized parameters"""
        self.params = self._copy_params(OPTIMIZED_PARAMS)
        self._update_all_ui()
        self.status_var.set("⭐ Loaded optimized parameters")
    
    def _update_all_ui(self):
        """Update all UI elements from params"""
        # Update sliders
        for key, data in self.slider_vars.items():
            channel_idx = data['channel_idx']
            if channel_idx is not None:
                param_name = key.split('_')[0]
                value = self.params['channels'][channel_idx][param_name]
            else:
                value = self.params[key]
            
            data['slider'].set(value)
            data['val_var'].set(f"{value:.3f}")
        
        # Update checkboxes
        for idx, var in self.check_vars.items():
            var.set(self.params['channels'][idx]['enabled'])
        
        # Update harmonic combos
        for idx, combo_var in self.combo_vars.items():
            harm_idx = self.params['channels'][idx]['harm_idx']
            for h_idx, h_name in AVAILABLE_HARMONICS:
                if h_idx == harm_idx:
                    combo_var.set(f"{h_idx}: {h_name}")
                    break
            
            # Update period label
            period_days = int(1460 / harm_idx)
            period_years = period_days / 365.25
            self.period_labels[idx].set(f"T≈{period_days}d ({period_years:.1f}y)")
        
        self.update_plot()
    
    def copy_params(self):
        """Copy current parameters to clipboard"""
        text = "=" * 50 + "\n"
        text += "PERRENOD WAVE MIXER - PARAMETERS\n"
        text += "=" * 50 + "\n\n"
        
        text += f"α (alpha) = {self.params['alpha']:.6f}\n"
        text += f"c (intercept) = {self.params['c']:.6f}\n\n"
        
        text += "CHANNELS:\n"
        for i, ch in enumerate(self.params['channels']):
            harm_idx = ch['harm_idx']
            period_days = int(1460 / harm_idx)
            status = "✓ ENABLED" if ch['enabled'] else "✗ disabled"
            text += f"\n  Channel {i+1} (h={harm_idx}, ~{period_days}d) - {status}\n"
            text += f"    A (amplitude) = {ch['A']:.6f}\n"
            text += f"    β (decay)     = {ch['beta']:.6f}\n"
            text += f"    φ (phase)     = {ch['phi']:.6f}\n"
        
        text += "\n" + "=" * 50 + "\n"
        text += "FORMULA:\n"
        text += f"ln(P) = {self.params['alpha']:.4f}·ln(t) + ({self.params['c']:.4f})\n"
        
        for ch in self.params['channels']:
            if ch['enabled']:
                h = ch['harm_idx']
                text += f"      + {ch['A']:.4f}·t^({ch['beta']:.4f})·cos({h}·{OMEGA:.4f}·ln(t) + {ch['phi']:.4f})\n"
        
        text += "=" * 50 + "\n"
        
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_var.set("📋 Parameters copied to clipboard!")
        except:
            self.status_var.set("Error copying to clipboard")
    
    def optimize_from_current(self):
        """Run optimization starting from current parameters"""
        self.status_var.set("⚡ Optimizing... please wait")
        self.root.update()
        
        # Run in thread
        thread = threading.Thread(target=self._run_optimization)
        thread.daemon = True
        thread.start()
    
    def _run_optimization(self):
        """Run the optimization"""
        try:
            # Get enabled channels
            enabled_channels = [(i, ch) for i, ch in enumerate(self.params['channels']) if ch['enabled']]
            
            if not enabled_channels:
                self.root.after(0, lambda: self.status_var.set("Enable at least one channel!"))
                return
            
            # Build bounds and x0
            bounds = [BOUNDS['alpha'], BOUNDS['c']]
            x0 = [self.params['alpha'], self.params['c']]
            
            for i, ch in enabled_channels:
                bounds.extend([BOUNDS['A'], BOUNDS['beta'], BOUNDS['phi']])
                x0.extend([ch['A'], ch['beta'], ch['phi']])
            
            # Objective function
            def objective(x):
                alpha, c = x[0], x[1]
                temp_channels = []
                for j, (i, ch) in enumerate(enabled_channels):
                    temp_channels.append({
                        'harm_idx': ch['harm_idx'],
                        'A': x[2 + j*3],
                        'beta': x[3 + j*3],
                        'phi': x[4 + j*3],
                        'enabled': True
                    })
                
                y_pred = perrenod_model(self.t_full, alpha, c, temp_channels)
                return np.sum((self.y_full - y_pred) ** 2)
            
            # Run optimization
            result = differential_evolution(objective, bounds, x0=x0, 
                                            maxiter=500, popsize=15,
                                            workers=1, updating='immediate',
                                            tol=1e-6, seed=42, polish=True)
            
            # Update parameters with results
            self.params['alpha'] = result.x[0]
            self.params['c'] = result.x[1]
            
            for j, (i, ch) in enumerate(enabled_channels):
                self.params['channels'][i]['A'] = result.x[2 + j*3]
                self.params['channels'][i]['beta'] = result.x[3 + j*3]
                self.params['channels'][i]['phi'] = result.x[4 + j*3]
            
            # Update UI on main thread
            self.root.after(0, self._update_after_optimization)
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)[:40]}"))
    
    def _update_after_optimization(self):
        """Update UI after optimization completes"""
        self._update_all_ui()
        self.status_var.set("✅ Optimization complete!")

# ============================================================
# MAIN
# ============================================================

def main():
    root = tk.Tk()
    
    # Style
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TFrame', background='#1a1a2e')
    style.configure('TLabelframe', background='#2d2d44')
    style.configure('TLabelframe.Label', background='#2d2d44', foreground='white')
    style.configure('TButton', font=('Arial', 9))
    style.configure('TCheckbutton', background='#2d2d44', foreground='white')
    style.configure('TScale', background='#2d2d44')
    
    app = WaveMixer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
