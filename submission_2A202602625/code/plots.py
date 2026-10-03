"""Đường cong train/val, metric và gradient; đóng figure sau khi lưu."""
from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result, path):
    c, h, s = result["cfg"], result["history"], result["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(h["epoch"], h["train_loss"], label="Train (eval, fixed subset)")
    axes[0].plot(h["epoch"], h["val_loss"], label="Validation")
    axes[0].set_ylabel("CE" if c["loss"] == "ce" else "MSE on logits (mean B*7)")
    axes[1].plot(h["epoch"], h["val_acc"], label="Val accuracy")
    axes[1].plot(h["epoch"], h["val_macro_f1"], label="Val macro-F1")
    axes[1].set(ylabel="Score", ylim=(0, 1))
    axes[2].plot(h["epoch"], h["grad_norm"], label="Mean L2 before clip")
    axes[2].plot(h["epoch"], h["grad_norm_max"], alpha=.6, label="Max L2 before clip")
    axes[2].set_ylabel("Gradient norm")
    if c["clip_norm"] is not None:
        axes[2].axhline(c["clip_norm"], linestyle=":", label="Clip threshold")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        if s.get("best_epoch", 0):
            ax.axvline(s["best_epoch"], color="gray", linestyle="--", alpha=.5)
    fig.suptitle(f"{c['exp_id']}: {c['optimizer']} lr={c['lr']} batch={c['batch']} "
        f"hidden={c['hidden']} dropout={c['dropout']} {c['init']} {c['precision']} [{s.get('status', '')}]")
    if s.get("reason"):
        axes[0].text(.05, .5, s["reason"], transform=axes[0].transAxes, wrap=True)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results, metric, path, title=""):
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        if r["history"]["epoch"]:
            ax.plot(r["history"]["epoch"], r["history"][metric], label=r["cfg"]["exp_id"])
    ax.set(xlabel="Epoch", ylabel=metric, title=title or metric)
    ax.grid(alpha=.25)
    if ax.lines:
        ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
