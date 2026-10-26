"""PINN-UTCI Valle de Aburrá — Interfaz web interactiva.

Uso:
    streamlit run app.py
"""

import json
from pathlib import Path

import folium
from streamlit_folium import st_folium
import base64
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─── Configuración de página ───
st.set_page_config(
    page_title="PINN-UTCI · Valle de Aburrá",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data" / "aoi"

SEASONS = ["full", "prune_bottom", "no_meteo", "minimal"]
SEASON_NAMES = {
    "full":         "Full (14 features)",
    "prune_bottom": "Pruned (8 features)",
    "no_meteo":     "No-meteo (9 features)",
    "minimal":      "Minimal (5 features)",
}
SEASON_COLORS = {
    "full":         "#1f77b4",
    "prune_bottom": "#2ca02c",
    "no_meteo":     "#ff7f0e",
    "minimal":      "#d62728",
}
CLIMATE_SEASONS = {
    "dry":  "Seca (dic-feb)",
    "wet":  "Húmeda (mar-may)",
    "dry2": "Veranillo (jun-jul)",
    "wet2": "Húmeda 2 (ago-nov)",
}


# ═══════════════════════════════════════════════════════════
# Cache de datos
# ═══════════════════════════════════════════════════════════

@st.cache_data
def load_metrics():
    """Carga todos los metrics_{name}.json disponibles."""
    out = {}
    for name in SEASONS:
        for fname in [f"metrics_{name}.json", "metrics.json"]:
            p = MODELS_DIR / fname
            if p.exists():
                out[name] = json.loads(p.read_text())
                break
    return out


@st.cache_data
def load_history():
    """Carga todos los history_{name}.json disponibles."""
    out = {}
    for name in SEASONS:
        for fname in [f"history_{name}.json", "history.json"]:
            p = MODELS_DIR / fname
            if p.exists():
                out[name] = json.loads(p.read_text())
                break
    return out


@st.cache_data
def load_sensitivity():
    """Carga el análisis de sensibilidad si existe."""
    p = MODELS_DIR / "sensitivity_spatial.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


# ═══════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🌡️ PINN-UTCI")
    st.markdown("**Valle de Aburrá · Colombia**")
    st.markdown("---")
    st.caption("Physics-Informed Neural Network para estimación de estrés térmico urbano")

    page = st.radio(
        "Navegación",
        [
            "🏠 Inicio",
            "🗺️ Mapas UTCI",
            "📊 Resultados de ablación",
            "🔬 Sensibilidad de features",
            "ℹ️ Metodología y citación",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Estado del proyecto**")
    metrics = load_metrics()
    st.metric("Modelos disponibles", f"{len(metrics)}/4")
    st.metric("Última actualización", "Oct 2026")

    st.markdown("---")
    st.caption("🔓 Código abierto (MIT)")
    st.caption("📧 manuel9519@hotmail.com")


# ═══════════════════════════════════════════════════════════
# Página: Inicio
# ═══════════════════════════════════════════════════════════

def page_home():
    st.title("🌡️ PINN-UTCI Valle de Aburrá")
    st.markdown(
        "**Estimación de estrés térmico urbano con Physics-Informed Neural Networks**"
    )
    st.markdown(
        "Modelo a 30 m de resolución espacial · ciclo diurno completo · "
        "4 temporadas climáticas · validación con holdout espacial."
    )
    st.markdown("---")

    metrics = load_metrics()

    # ─── Métricas principales ───
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if "full" in metrics:
            m = metrics["full"]
            st.metric(
                "MAE (full)",
                f"{m['test_mae']:.3f} °C",
                delta=f"R² = {m['test_r2']:.4f}",
                delta_color="off",
            )
    with col2:
        if "full" in metrics:
            st.metric("RMSE (full)", f"{metrics['full']['test_rmse']:.3f} °C")
    with col3:
        if "minimal" in metrics:
            st.metric(
                "MAE (minimal)",
                f"{metrics['minimal']['test_mae']:.3f} °C",
                delta=f"R² = {metrics['minimal']['test_r2']:.4f}",
                delta_color="off",
            )
    with col4:
        if "minimal" in metrics:
            st.metric("RMSE (minimal)", f"{metrics['minimal']['test_rmse']:.3f} °C")

    st.markdown("---")

    # ─── Panel de temporadas ───
    st.subheader("UTCI predicho por temporada — 15:00 hora local")
    panel_path = OUTPUTS_DIR / "seasonal_panel.png"
    if panel_path.exists():
        st.image(str(panel_path), use_container_width=True)
        st.caption(
            "🔴 Rojo indica mayor estrés térmico · 🔵 Azul indica menor estrés. "
            "La cobertura reducida en temporadas húmedas se debe a nubosidad en Landsat."
        )
    else:
        st.warning(
            "No se encontró `seasonal_panel.png`. Genera los mapas con:\n\n"
            "```python -m src.pinn.plot_seasonal_panel```"
        )

    st.markdown("---")

    # ─── Qué ofrece esta app ───
    st.subheader("¿Qué puedes hacer con esta herramienta?")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("### 🗺️ Explorar mapas")
        st.markdown(
            "Consulta el UTCI predicho para cualquier temporada y hora del día."
        )
    with cols[1]:
        st.markdown("### 📊 Comparar modelos")
        st.markdown(
            "Compara 4 configuraciones con diferentes números de features."
        )
    with cols[2]:
        st.markdown("### 🔬 Analizar sensibilidad")
        st.markdown(
            "Identifica qué variables son críticas para el modelo."
        )

    st.markdown("---")
    st.info(
        "📌 **Alcance del modelo:** Entrenado para el Valle de Aburrá en 2024. "
        "No es transferible directamente a otras regiones o años sin reentrenamiento. "
        "Ver sección *Metodología y citación* para detalles."
    )


# ═══════════════════════════════════════════════════════════
# Página: Mapas
# ═══════════════════════════════════════════════════════════


import geopandas as gpd
from pyproj import Transformer


@st.cache_data
def load_overlay_meta(season, hour):
    json_path = OUTPUTS_DIR / "overlays" / f"overlay_{season}_{hour:02d}h.json"
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text())


@st.cache_data
def load_municipios():
    path = PROJECT_ROOT / "data" / "aoi" / "valle_aburra.geojson"
    if not path.exists():
        return None
    return gpd.read_file(path).to_crs("EPSG:4326")


@st.cache_data

def get_raster_info(season, hour=12):
    """Obtiene el centro del AOI desde la metadata del overlay (sin .nc)."""
    meta = load_overlay_meta(season, hour)
    if meta is None:
        # Fallback: buscar cualquier overlay de esa temporada
        for h in range(24):
            meta = load_overlay_meta(season, h)
            if meta is not None:
                break

    if meta is None:
        return None

    bounds = meta["bounds"]
    center = [
        (bounds[0][0] + bounds[1][0]) / 2,
        (bounds[0][1] + bounds[1][1]) / 2,
    ]
    return {"center": center}

def png_to_data_uri(path):
    """Convierte un PNG local a data URI base64 para Folium."""
    import base64
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def detect_name_column(gdf):
    """Detecta la columna con el nombre del municipio."""
    candidates = [
        "S_Municipi",
        "NAME_2",
        "NOMBRE", "nombre", "Nombre",
        "MUNICIPIO", "municipio", "Municipio",
        "name", "NAME", "Name",
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    for c in gdf.columns:
        if c == "geometry":
            continue
        if gdf[c].dtype == object and not gdf[c].astype(str).str.match(r"^\d+$").all():
            return c
    return None

def page_maps():
    st.title("🗺️ Mapas UTCI interactivos")
    st.markdown("Navega sobre el Valle de Aburrá y activa capas a tu gusto.")
    st.markdown("---")

    # ─── Fila 1: controles principales ───
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        climate_season = st.selectbox(
            "Temporada climática",
            options=list(CLIMATE_SEASONS.keys()),
            format_func=lambda k: CLIMATE_SEASONS[k],
        )
    with col2:
        hour = st.slider("Hora local", 0, 23, 15, format="%d:00")
    with col3:
        BASE_MAPS = {
            "OpenStreetMaps": {
                "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
            },
            "Oscuro": {
                "url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                "attr": '&copy; <a href="https://carto.com/attributions">CARTO</a>',
            },
            "Claro": {
                "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                "attr": '&copy; <a href="https://carto.com/attributions">CARTO</a>',
            },
            "Satélite": {
                "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                "attr": "Tiles &copy; Esri",
            },
            "Terreno": {
                "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                "attr": "Tiles &copy; Esri",
            },
        }
        base_choice = st.selectbox(
            "Estilo de mapa base",
            options=list(BASE_MAPS.keys()),
            index=0,
        )

    # ─── Fila 2: capas y opacidad ───
    col4, col5, col6, col7 = st.columns([1, 2, 1, 1])
    with col4:
        show_utci = st.checkbox("UTCI", value=True)
    with col5:
        utci_opacity = st.slider(
            "Opacidad UTCI",
            min_value=0.0, max_value=1.0, value=0.80, step=0.05,
            disabled=not show_utci,
        )
    with col6:
        show_municipios = st.checkbox("Municipios", value=True)
    with col7:
        show_labels = st.checkbox("Etiquetas", value=True,
                                  disabled=not show_municipios)

    st.markdown("---")

    # ─── Cargar datos ───
    overlay_png = OUTPUTS_DIR / "overlays" / f"overlay_{climate_season}_{hour:02d}h.png"
    overlay_meta = load_overlay_meta(climate_season, hour)
    raster_info = get_raster_info(climate_season, hour)

    if overlay_meta is None or not overlay_png.exists():
        st.warning(
            f"No existe el overlay para **{CLIMATE_SEASONS[climate_season]}** a las "
            f"**{hour}:00**.\n\nGeneralo con:\n\n"
            f"```python -m src.pinn.generate_overlay --season {climate_season} --hour {hour}```"
        )
        return

    if raster_info is None:
        st.error("No se pudo determinar el centro del AOI.")
        return

    gdf = load_municipios()

    # ─── Construir mapa ───
    m = folium.Map(
        location=raster_info["center"],
        zoom_start=12,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles=BASE_MAPS[base_choice]["url"],
        attr=BASE_MAPS[base_choice]["attr"],
        name=base_choice,
        max_zoom=19,
    ).add_to(m)

    # ─── Overlay UTCI ───
    if show_utci:
        folium.raster_layers.ImageOverlay(
            image=png_to_data_uri(overlay_png),
            bounds=overlay_meta["bounds"],
            opacity=utci_opacity,
            name="UTCI predicho (PINN)",
            interactive=False,
        ).add_to(m)

    # ─── Municipios + etiquetas ───
    if show_municipios and gdf is not None:
        name_col = detect_name_column(gdf)
        tooltip = folium.GeoJsonTooltip(
            fields=[name_col], aliases=["Municipio:"], sticky=True
        ) if name_col else None

        folium.GeoJson(
            gdf.__geo_interface__,
            name="Municipios (Valle de Aburrá)",
            style_function=lambda x: {
                "fillColor": "#ffffff", "color": "#ffffff",
                "weight": 1.5, "fillOpacity": 0.02,
            },
            highlight_function=lambda x: {
                "fillColor": "#ffff00", "color": "#ffcc00",
                "weight": 2.5, "fillOpacity": 0.15,
            },
            tooltip=tooltip,
        ).add_to(m)

        if show_labels and name_col:
            labels_layer = folium.FeatureGroup(name="Etiquetas municipios", show=True)

            for _, row in gdf.iterrows():
                geom = row["geometry"]
                centroid = geom.centroid
                label = str(row[name_col]).strip().title()

                folium.Marker(
                    location=[centroid.y, centroid.x],
                    icon=folium.DivIcon(
                        html=f'''
                        <div style="
                            font-family: 'Inter', Arial, sans-serif;
                            font-size: 11px;
                            font-weight: 700;
                            color: #ffffff;
                            background: rgba(0, 0, 0, 0.78);
                            padding: 3px 9px;
                            border-radius: 10px;
                            text-align: center;
                            white-space: nowrap;
                            border: 1px solid rgba(255,255,255,0.35);
                            box-shadow: 0 2px 4px rgba(0,0,0,0.5);
                            pointer-events: none;
                        ">{label}</div>
                        ''',
                        icon_size=(120, 22),
                        icon_anchor=(60, 11),
                    ),
                ).add_to(labels_layer)

            labels_layer.add_to(m)

    # ─── Leyenda vertical personalizada (HTML/CSS) ───
    if show_utci:
        vmin = overlay_meta['vmin']
        vmax = overlay_meta['vmax']
        vmid = (vmin + vmax) / 2

        # Generamos la leyenda HTML inyectada directamente
        legend_html = f'''
        <div style="
            position: absolute;
            bottom: 50px;
            right: 20px;
            width: 70px;
            height: 280px;
            background-color: white;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 10px 5px;
            z-index: 9999;
            font-family: 'Inter', Arial, sans-serif;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: space-between;
        ">
            <div style="font-size: 11px; font-weight: 700; color: #000; text-align: center; margin-bottom: 5px;">
                UTCI<br>(°C)
            </div>
            <div style="font-size: 13px; font-weight: 700; color: #000;">{vmax:.1f}</div>
            <div style="
                flex-grow: 1;
                width: 18px;
                margin: 6px 0;
                background: linear-gradient(to bottom, 
                    #a50026, #d73027, #f46d43, #fdae61, #fee090, 
                    #ffffbf, #e0f3f8, #abd9e9, #74add1, #4575b4, #313695
                );
                border-radius: 3px;
                border: 1px solid #666;
            "></div>
            <div style="font-size: 13px; font-weight: 700; color: #000;">{vmin:.1f}</div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

    # ─── Control de capas colapsado ───
    folium.LayerControl(collapsed=True).add_to(m)

    # ─── Render ───
    st_folium(m, height=650, use_container_width=True, returned_objects=[])

    st.caption(
        f"**{CLIMATE_SEASONS[climate_season]}** · **{hour}:00** local · "
        f"Rango UTCI: **{overlay_meta['vmin']:.1f} °C** a **{overlay_meta['vmax']:.1f} °C** "
        f"· Opacidad: {utci_opacity:.0%}"
    )

    st.markdown("---")
    st.subheader("Vista comparativa de las 4 temporadas")
    panel_path = OUTPUTS_DIR / "seasonal_panel.png"
    if panel_path.exists():
        st.image(str(panel_path), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# Página: Resultados de ablación
# ═══════════════════════════════════════════════════════════

def page_ablation():
    st.title("📊 Resultados de ablación")
    st.markdown(
        "Comparación sistemática de 4 configuraciones con diferentes números de features."
    )
    st.markdown("---")

    metrics = load_metrics()
    if not metrics:
        st.error("No hay métricas disponibles. Entrena al menos un modelo.")
        return

    # ─── Tabla ───
    st.subheader("Tabla comparativa")

    import pandas as pd
    rows = []
    for name in SEASONS:
        if name not in metrics:
            continue
        m = metrics[name]
        rows.append({
            "Configuración": SEASON_NAMES[name],
            "n_features": m.get("in_dim", 0) - 4,
            "MAE (°C)": f"{m['test_mae']:.3f}",
            "RMSE (°C)": f"{m['test_rmse']:.3f}",
            "R²": f"{m['test_r2']:.4f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ─── Bar chart interactivo ───
    st.subheader("Comparación visual")

    chart_data = []
    for name in SEASONS:
        if name not in metrics:
            continue
        m = metrics[name]
        chart_data.append({
            "Config": SEASON_NAMES[name].split(" ")[0],
            "Métrica": "MAE",
            "Valor (°C)": m["test_mae"],
            "color": SEASON_COLORS[name],
        })
        chart_data.append({
            "Config": SEASON_NAMES[name].split(" ")[0],
            "Métrica": "RMSE",
            "Valor (°C)": m["test_rmse"],
            "color": SEASON_COLORS[name],
        })

    fig = go.Figure()
    for metric_name in ["MAE", "RMSE"]:
        subset = [d for d in chart_data if d["Métrica"] == metric_name]
        fig.add_trace(go.Bar(
            name=metric_name,
            x=[d["Config"] for d in subset],
            y=[d["Valor (°C)"] for d in subset],
            marker_color=[d["color"] for d in subset],
            text=[f"{d['Valor (°C)']:.3f}" for d in subset],
            textposition="outside",
            opacity=1.0 if metric_name == "MAE" else 0.6,
        ))
    fig.update_layout(
        barmode="group",
        yaxis_type="log",
        yaxis_title="Error (°C)",
        xaxis_title="Configuración",
        height=500,
        legend_title="Métrica",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ─── Curvas de entrenamiento ───
    st.subheader("Curvas de entrenamiento")
    history = load_history()
    if history:
        fig = go.Figure()
        for name in SEASONS:
            if name not in history:
                continue
            h = history[name]
            epochs = []
            maes = []
            for stage in ["stage1", "stage2"]:
                for item in h.get(stage, []):
                    epochs.append(item["epoch"])
                    maes.append(item["val_mae"])
            fig.add_trace(go.Scatter(
                x=epochs, y=maes,
                mode="lines",
                name=SEASON_NAMES[name],
                line=dict(color=SEASON_COLORS[name], width=2),
            ))
        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Validation MAE (°C)",
            yaxis_type="log",
            height=500,
            hovermode="x unified",
        )
        fig.add_vline(x=250, line_dash="dot", line_color="gray",
                      annotation_text="Etapa 2 →", annotation_position="top")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay curvas de entrenamiento disponibles.")

    st.markdown("---")

    # ─── Hallazgo principal ───
    st.subheader("🔬 El hallazgo principal: 'physics residual floor'")
    st.markdown("""
    El término físico **mejora** la predicción cuando el modelo tiene 
    acceso a la meteorología completa, pero la **degrada** cuando esas 
    variables se eliminan.

    Existe un **piso irreducible** del residuo físico que depende de las 
    features disponibles. Cuando ese piso es alto (por falta de información), 
    el optimizador distorsiona la predicción para intentar reducirlo, 
    sacrificando precisión.

    **Implicación:** la física no siempre ayuda. Requiere coherencia 
    informacional con la ecuación gobernante.
    """)


# ═══════════════════════════════════════════════════════════
# Página: Sensibilidad
# ═══════════════════════════════════════════════════════════

def page_sensitivity():
    st.title("🔬 Sensibilidad de features")
    st.markdown(
        "Ranking de importancia de las 14 covariables usando dos métodos complementarios."
    )
    st.markdown("---")

    sens = load_sensitivity()
    if sens is None:
        st.warning(
            "No se encontró `sensitivity_spatial.json`. "
            "Ejecuta el análisis:\n\n"
            "```python -m src.run_sensitivity --split-mode spatial --by-season --permutation```"
        )

        # Fallback: tabla estática con los datos conocidos
        st.subheader("Ranking de sensibilidad (datos de referencia)")
        static_data = [
            ("SSRD", 27.64, 0.0506, "Gate"),
            ("MRT", 10.45, 1.9012, "Critical"),
            ("elevation", 9.87, 0.0015, "Redundant"),
            ("RH", 8.10, 0.6134, "Important"),
            ("T_air", 7.17, 2.6643, "Critical"),
            ("aspect", 1.56, 0.0006, "Marginal"),
            ("LST", 0.27, 0.0162, "Marginal"),
            ("slope", 0.17, 0.0017, "Marginal"),
            ("wind_speed", 0.06, 0.1452, "Moderate"),
            ("landcover", 0.06, 0.0015, "Null"),
            ("FVC", 0.03, 0.0374, "Null"),
            ("NDVI", 0.02, 0.0470, "Null"),
            ("ALBEDO", 0.0008, 0.0022, "Null"),
            ("EMIS", 0.0004, 0.0038, "Null"),
        ]

        import pandas as pd
        df = pd.DataFrame(static_data, columns=["Feature", "Sensibilidad", "ΔMAE (°C)", "Categoría"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = go.Figure(go.Bar(
            x=df["Sensibilidad"],
            y=df["Feature"],
            orientation="h",
            marker_color=[
                "#d62728" if c == "Critical" else
                "#ff7f0e" if c == "Important" else
                "#1f77b4" if c in ("Gate", "Moderate") else
                "#cccccc"
                for c in df["Categoría"]
            ],
            text=df["Sensibilidad"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Sensibilidad (media |∂ŷ/∂xᵢ|)",
            yaxis_title="Feature",
            yaxis=dict(autorange="reversed"),
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Interpretación")
        st.markdown("""
        | Categoría | Features | Interpretación |
        |---|---|---|
        | **Critical** | T_air, MRT | Son los 4 inputs de la fórmula UTCI. Impacto directo. |
        | **Important** | RH | Contribuye a la humedad relativa. |
        | **Moderate** | wind_speed | Impacto moderado pero importante. |
        | **Gate** | SSRD | Alta pendiente local, bajo impacto global. |
        | **Redundant** | elevation | Reemplazable por x,y. |
        | **Marginal / Null** | LST, slope, aspect, NDVI, FVC, ALBEDO, EMIS, landcover | Contribuyen poco o son redundantes con LST. |
        """)
        return

    # Si existe el JSON, mostrar dinámicamente
    st.success("Análisis de sensibilidad cargado.")
    st.json(sens)


# ═══════════════════════════════════════════════════════════
# Página: Metodología
# ═══════════════════════════════════════════════════════════

def page_methodology():
    st.title("ℹ️ Metodología y citación")
    st.markdown("---")

    st.subheader("🎯 ¿Qué hace este modelo?")
    st.markdown("""
    Estima el **Universal Thermal Climate Index (UTCI)** sobre el Valle de Aburrá, 
    Colombia, a **30 m de resolución espacial** y **resolución horaria**, 
    combinando datos satelitales, reanálisis meteorológico y leyes físicas.
    """)

    st.subheader("📐 Arquitectura")
    st.markdown("""
    - **Modelo:** MLP con 4 capas ocultas, 128 neuronas, activación `tanh`
    - **Entrada:** 18 dimensiones (4 coordenadas + 14 features)
    - **Salida:** 1 valor (UTCI en °C)
    - **Pérdida:** datos ponderados + residuo PDE + suavizado temporal
    """)

    st.subheader("🗺️ Datos de entrada")
    st.markdown("""
    | Fuente | Variables | Resolución |
    |---|---|---|
    | Landsat 8/9 | LST, NDVI, FVC, albedo, emisividad | 30 m |
    | SRTM | Elevación, pendiente, aspecto | 30 m |
    | Dynamic World | Cobertura del suelo | 10 m |
    | ERA5-Land | T_air, RH, viento, radiación | 9 km |
    """)

    st.subheader("✅ Validación")
    st.markdown("""
    - **Holdout espacial:** K-means (k=5) sobre coordenadas UTM. 3 clusters para 
      entrenamiento, 1 para validación, 1 para test.
    - **Ablación:** 4 configuraciones con distintos números de features.
    - **Sensibilidad:** gradiente + permutación (Breiman 2001).
    """)

    st.markdown("---")

    st.subheader("⚠️ Limitaciones")
    st.markdown("""
    - **Cobertura nubosa:** En temporada húmeda solo ~9% de píxeles son válidos 
      (vs ~30% en seca).
    - **Un solo año:** 2024. No captura variabilidad interanual (El Niño / La Niña).
    - **MRT simplificada:** Asume `F_sky = 0.5` constante, ignorando obstrucción urbana.
    - **Sin validación in-situ:** Pendiente integrar SIATA.
    - **Alcance geográfico:** Específico al Valle de Aburrá. No transferible a otras 
      regiones sin reentrenamiento.
    """)

    st.markdown("---")

    st.subheader("📖 Cómo citar")
    st.code("""@software{luna2026pinn,
  author       = {Luna, Juan Manuel},
  title        = {PINN-UTCI Valle de Aburrá: Physics-Informed Neural
                  Network for Urban Thermal Stress Estimation},
  year         = {2026},
  version      = {1.0.0},
  license      = {MIT},
  url          = {https://github.com/tu-usuario/pinn-utci-aburra}
}""", language="bibtex")

    st.subheader("📄 Licencia")
    st.markdown("""
    - **Código fuente:** MIT (libre, incluso comercial, con atribución)
    - **Datos derivados y modelo:** CC-BY-4.0
    - **Servicios comerciales:** Contactar (ver `COMMERCIAL.md`)
    """)

    st.subheader("📬 Contacto")
    st.markdown("""
    **Juan Manuel Luna**  
    Investigador independiente · Medellín, Colombia  
    📧 manuel9519@hotmail.com  
    🐛 [GitHub Issues](https://github.com/tu-usuario/pinn-utci-aburra/issues)
    """)


# ═══════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════

if page == "🏠 Inicio":
    page_home()
elif page == "🗺️ Mapas UTCI":
    page_maps()
elif page == "📊 Resultados de ablación":
    page_ablation()
elif page == "🔬 Sensibilidad de features":
    page_sensitivity()
elif page == "ℹ️ Metodología y citación":
    page_methodology()