"""Crea/actualiza la carpeta de deploy con solo lo necesario para Streamlit.

Uso:
    python sync_to_deploy.py
"""

import shutil
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
DEST = SOURCE.parent / "pinn_utci_aburra_deploy"


def copy_file(rel_path):
    src = SOURCE / rel_path
    dst = DEST / rel_path
    if not src.exists():
        print(f"  [skip] {rel_path} (no existe)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  ✓ {rel_path}")


def copy_glob(rel_pattern):
    src_dir = SOURCE / Path(rel_pattern).parent
    dst_dir = DEST / Path(rel_pattern).parent
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.glob(Path(rel_pattern).name):
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
    count = len(list(src_dir.glob(Path(rel_pattern).name)))
    print(f"  ✓ {rel_pattern} ({count} archivos)")


def main():
    if DEST.exists():
        print(f"⚠️  {DEST} ya existe.")
        resp = input("¿Sobrescribir? (s/N): ").strip().lower()
        if resp != "s":
            print("Cancelado.")
            return
        shutil.rmtree(DEST)

    DEST.mkdir()
    print(f"\n📁 Creando {DEST.name}/...\n")

    # ─── Archivos raíz ───
    print("📄 Documentación y configuración:")
    for f in ["LICENSE", "LICENSE-DATA", "CITATION.cff", "COMMERCIAL.md"]:
        copy_file(f)

    # ─── Configuración Streamlit ───
    print("\n⚙️  Configuración Streamlit:")
    (DEST / ".streamlit").mkdir(exist_ok=True)
    (DEST / ".streamlit" / "config.toml").write_text(
        '[theme]\n'
        'primaryColor = "#d62728"\n'
        'backgroundColor = "#ffffff"\n'
        'secondaryBackgroundColor = "#f0f2f6"\n'
        'textColor = "#262730"\n'
        'font = "sans serif"\n'
        '\n'
        '[server]\n'
        'maxUploadSize = 50\n'
        '\n'
        '[browser]\n'
        'gatherUsageStats = false\n',
        encoding="utf-8",
    )
    print("  ✓ .streamlit/config.toml")

    # ─── Requirements ───
    print("\n📦 Requirements:")
    (DEST / "requirements.txt").write_text(
        "streamlit>=1.30\n"
        "streamlit-folium>=0.18\n"
        "folium>=0.15\n"
        "plotly>=5.18\n"
        "pandas>=2.0\n"
        "numpy>=1.24\n"
        "matplotlib>=3.7\n"
        "geopandas>=0.14\n"
        "shapely>=2.0\n"
        "pyproj>=3.6\n"
        "branca>=0.7\n",
        encoding="utf-8",
    )
    print("  ✓ requirements.txt")

    # ─── .gitignore ───
    print("\n🚫 .gitignore:")
    (DEST / ".gitignore").write_text(
        "# Python\n"
        "__pycache__/\n"
        "*.pyc\n"
        ".venv/\n"
        "\n"
        "# Datos\n"
        "data/raw/\n"
        "data/processed/\n"
        "data/interim/\n"
        "\n"
        "# Modelos entrenados (no van al deploy)\n"
        "models/*.pt\n"
        "\n"
        "# IDE\n"
        ".vscode/\n"
        ".idea/\n"
        "\n"
        "# Sistema\n"
        ".DS_Store\n"
        "Thumbs.db\n",
        encoding="utf-8",
    )
    print("  ✓ .gitignore")

    # ─── GeoJSON ───
    print("\n🗺️  Datos geoespaciales:")
    copy_file("data/aoi/valle_aburra.geojson")

    # ─── Métricas ───
    print("\n📊 Métricas e historiales:")
    copy_glob("models/metrics_*.json")
    copy_glob("models/history_*.json")

    # ─── Outputs (mapas comparativos) ───
    print("\n🖼️  Outputs principales:")
    for f in [
        "seasonal_panel.png",
        "ablation_mae_rmse.png",
        "training_curves_mae.png",
        "training_curves_physics.png",
        "training_curves_stages.png",
    ]:
        copy_file(f"outputs/{f}")

    # ─── Overlays ───
    print("\n🗺️  Overlays (96 pares PNG+JSON):")
    copy_glob("outputs/overlays/*.png")
    copy_glob("outputs/overlays/*.json")

    # ─── App y README ───
    print("\n🖥️  Aplicación y README:")
    copy_file("app.py")
    copy_file("README.md")

    # ─── Tamaño final ───
    total = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"\n✅ Deploy listo: {DEST}")
    print(f"📦 Tamaño total: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()