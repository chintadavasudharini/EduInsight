"""
Page 10: Optimization and Training Lab (CO-4: Deep Learning & Optimization).
Dedicated PyTorch experimental lab:
- Multi-Optimizer Benchmark: SGD, Momentum, Nesterov, Adagrad, RMSprop, Adam, AdamW, Lion
- Learning Rate Schedulers: Constant, StepLR, CosineAnnealing, WarmRestarts, OneCycle
- Regularization Strategies: L2 (Weight Decay), Dropout, Batch Normalization
- Early Stopping Callback Demonstration
- Weight Initializations: Xavier (Glorot) vs He (Kaiming)
- Gradient Accumulation & Distributed Data Parallel (DDP) conceptualization
Runs entirely on CPU with real-time PyTorch training curves.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.utils.config import FEATURE_NAMES, TARGET_REGRESSION
from src.data.loader import load_splits
from src.optimization.optimizers_lab import run_optimizer_benchmark
from src.optimization.schedulers_lab import run_scheduler_benchmark
from src.optimization.regularization_lab import (
    compare_regularization_techniques,
    demonstrate_early_stopping,
    compare_initializations,
    demonstrate_gradient_accumulation,
)


def render_page():
    st.markdown('<div class="main-header">⚡ Optimization & Training Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">PyTorch deep learning optimizer comparisons, learning-rate schedules, and regularization techniques.</div>',
        unsafe_allow_html=True,
    )

    train_df, val_df, _ = load_splits()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_df[FEATURE_NAMES].values)
    y_tr = train_df[TARGET_REGRESSION].values
    X_va = scaler.transform(val_df[FEATURE_NAMES].values)
    y_va = val_df[TARGET_REGRESSION].values

    tab_opts, tab_sched, tab_reg, tab_init, tab_scale = st.tabs([
        "🚀 8 Optimizers Benchmark",
        "📈 Learning Rate Schedulers",
        "🛡️ Regularization & Early Stopping",
        "🎲 Xavier vs He Initialization",
        "🏗️ Gradient Accumulation & DDP",
    ])

    with tab_opts:
        st.subheader("Multi-Optimizer Convergence Comparison (CPU PyTorch)")
        st.markdown(
            "Comparing convergence speed and final validation loss across 8 first-order optimization algorithms "
            "on the StudentMLP regression architecture."
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            opts_to_run = st.multiselect(
                "Select Optimizers to Compare:",
                ["SGD", "Momentum", "Nesterov", "Adagrad", "RMSprop", "Adam", "AdamW", "Lion"],
                default=["SGD", "Momentum", "RMSprop", "Adam", "AdamW", "Lion"],
            )
            epochs_opt = st.slider("Training Epochs:", 15, 60, 30, 5)
            lr_opt = st.slider("Learning Rate:", 0.001, 0.05, 0.01, 0.002)

            run_opt_btn = st.button("🚀 Train & Compare Optimizers", use_container_width=True)

        with col2:
            if run_opt_btn or "opt_results" not in st.session_state:
                with st.spinner("Training PyTorch networks with selected optimizers..."):
                    opt_res = run_optimizer_benchmark(
                        X_tr, y_tr, X_va, y_va,
                        optimizer_list=opts_to_run,
                        epochs=epochs_opt,
                        lr=lr_opt,
                    )
                    st.session_state["opt_results"] = opt_res

            res = st.session_state.get("opt_results", {})

            if res:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
                for opt_name, data in res.items():
                    ax1.plot(data["train_losses"], label=opt_name, linewidth=1.8)
                    ax2.plot(data["val_losses"], label=opt_name, linewidth=1.8)

                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Train MSE Loss")
                ax1.set_title("Training Loss Convergence", fontsize=10, fontweight="bold")
                ax1.legend()
                ax1.grid(True, linestyle=":", alpha=0.5)

                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Validation MSE Loss")
                ax2.set_title("Validation Loss Curve", fontsize=10, fontweight="bold")
                ax2.legend()
                ax2.grid(True, linestyle=":", alpha=0.5)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Summary Table
                comp_data = []
                for k, v in res.items():
                    comp_data.append({
                        "Optimizer": k,
                        "Final Train Loss (MSE)": round(v["train_losses"][-1], 2),
                        "Final Val Loss (MSE)": round(v["val_losses"][-1], 2),
                        "Final Val RMSE (Marks)": round(v["final_val_rmse"], 2),
                    })
                comp_df = pd.DataFrame(comp_data).sort_values(by="Final Val Loss (MSE)")
                st.dataframe(comp_df, use_container_width=True)

    with tab_sched:
        st.subheader("Dynamic Learning Rate Schedules")
        sched_res = run_scheduler_benchmark(X_tr, y_tr, X_va, y_va, epochs=30, base_lr=0.03)

        fig, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(10, 3.8))
        for s_name, data in sched_res.items():
            ax_s1.plot(data["lr_history"], label=s_name, linewidth=1.8)
            ax_s2.plot(data["val_losses"], label=s_name, linewidth=1.8)

        ax_s1.set_xlabel("Step / Epoch")
        ax_s1.set_ylabel("Learning Rate (η)")
        ax_s1.set_title("LR Schedule Decay Profile", fontsize=10, fontweight="bold")
        ax_s1.legend()
        ax_s1.grid(True, linestyle=":", alpha=0.5)

        ax_s2.set_xlabel("Epoch")
        ax_s2.set_ylabel("Validation MSE Loss")
        ax_s2.set_title("Validation Loss vs Schedule", fontsize=10, fontweight="bold")
        ax_s2.legend()
        ax_s2.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_reg:
        st.subheader("Regularization & Generalization Gap")
        reg_res = compare_regularization_techniques(X_tr, y_tr, X_va, y_va, epochs=35)

        fig, ax = plt.subplots(figsize=(7, 3.8))
        for reg_name, data in reg_res.items():
            ax.plot(data["val_losses"], label=f"{reg_name} (Gap: {data['generalization_gap']:.1f})", linewidth=1.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE Loss")
        ax.set_title("Regularization Generalization Comparison", fontsize=11, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("##### Early Stopping Callback")
        patience_v = st.slider("Early Stopping Patience (epochs):", 2, 10, 5)
        es_res = demonstrate_early_stopping(X_tr, y_tr, X_va, y_va, patience=patience_v)

        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("Stopped Early?", "Yes" if es_res["stopped_early"] else "No (Max Epochs)")
        with e2:
            st.metric("Halt Epoch", es_res["stopped_epoch"])
        with e3:
            st.metric("Best Restored Epoch", es_res["best_epoch"])

    with tab_init:
        st.subheader("Weight Initialization: Xavier vs He (Kaiming)")
        init_res = compare_initializations(X_tr, y_tr, X_va, y_va, epochs=25)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(init_res["xavier"]["train_losses"], label="Xavier Uniform", color="#2563EB", linewidth=2)
        ax.plot(init_res["he"]["train_losses"], label="He Kaiming Normal (for ReLU)", color="#10B981", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.set_title("Initial Loss Trajectory: Xavier vs He", fontsize=10, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_scale:
        st.subheader("Gradient Accumulation & Distributed Data Parallel (DDP)")
        grad_acc = demonstrate_gradient_accumulation(X_tr, y_tr, effective_batch_size=64, micro_batch_size=16)

        st.info(grad_acc["explanation"])

        st.markdown("##### Distributed Data Parallel (DDP) Architecture")
        st.markdown("""
        In PyTorch DDP:
        1. Model parameters are replicated identically across all GPU/CPU worker processes.
        2. Each process consumes an independent non-overlapping mini-batch partition via `DistributedSampler`.
        3. During `loss.backward()`, process gradients are asynchronously synchronized across processes using `AllReduce` rings.
        4. Optimization parameters update simultaneously, maintaining identical model states with zero parameter server bottlenecks.
        """)


if __name__ == "__main__":
    render_page()
