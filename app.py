import numpy as np
import joblib
import os
import pandas as pd
from datetime import datetime, timedelta
from pytrends.request import TrendReq

class ModeloPrecios:
    def __init__(self, alpha=0.1, gamma=0.9):
        self.alpha = alpha
        self.gamma = gamma
        self.precios_posibles = np.arange(40, 101, 5)  # 40, 45, ..., 100
        self.estado_dims = (3, 2, 2)  # (disponibilidad, competidor, tendencia)
        self.q_table = self._cargar_q_table()

    def _cargar_q_table(self):
        if os.path.exists("q_table.pkl"):
            return joblib.load("q_table.pkl")
        shape = self.estado_dims + (len(self.precios_posibles),)
        return np.zeros(shape)

    def _guardar_q_table(self):
        joblib.dump(self.q_table, "q_table.pkl")

    def obtener_o_actualizar_tendencia(self):
        pytrends = TrendReq(hl='es-ES', tz=360)
        kw_list = ["alquiler de furgonetas"]
        hoy = datetime.today()
        hace_un_ano = hoy - timedelta(days=365)
        timeframe = f"{hace_un_ano.strftime('%Y-%m-%d')} {hoy.strftime('%Y-%m-%d')}"
        pytrends.build_payload(kw_list, cat=478, timeframe=timeframe, geo='ES')
        df_trends = pytrends.interest_over_time()
        if df_trends.empty:
            raise ValueError("No se pudieron obtener datos de Google Trends.")
        df_trends = df_trends.reset_index()
        df_cache = pd.DataFrame({
            "fecha_semanal": df_trends["date"].dt.strftime("%Y-%m-%d"),
            "indice_valor": df_trends[kw_list[0]].astype(int)
        })
        df_cache.to_csv("google_trends_cache.csv", index=False)
        return int(df_cache["indice_valor"].iloc[-1])

    def _discretizar_estado(self, disponibilidad, precio_competidor, indice_google_trends):
        if disponibilidad < 3:
            d = 0
        elif disponibilidad <= 6:
            d = 1
        else:
            d = 2
        c = 0 if precio_competidor < 60 else 1
        t = 0 if indice_google_trends < 50 else 1
        return (d, c, t)

    def obtener_precio_optimo(self, estado_discretizado):
        d, c, t = estado_discretizado
        q_values = self.q_table[d, c, t]
        if np.all(q_values == 0):
            idx = np.random.choice(len(self.precios_posibles))
        else:
            idx = np.argmax(q_values)
        return self.precios_posibles[idx]

    def actualizar_q_table(self, estado_antiguo, accion_idx, recompensa, estado_nuevo):
        d_old, c_old, t_old = estado_antiguo
        d_new, c_new, t_new = estado_nuevo
        q_antiguo = self.q_table[d_old, c_old, t_old, accion_idx]
        q_max_nuevo = np.max(self.q_table[d_new, c_new, t_new])
        self.q_table[d_old, c_old, t_old, accion_idx] = (
            q_antiguo + self.alpha * (recompensa + self.gamma * q_max_nuevo - q_antiguo)
        )
        self._guardar_q_table()


def inicializar_historico():
    archivo = "historico_alquileres.csv"
    columnas = [
        "fecha", "disponibilidad_inicial", "reservas_concretadas",
        "precio_fijado", "ingreso_total", "precio_competidor", "indice_google_trends"
    ]
    if not os.path.exists(archivo) or os.stat(archivo).st_size < 60:
        fechas = [(datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(50, 0, -1)]
        datos = []
        for fecha in fechas:
            disponibilidad = np.random.randint(1, 11)
            reservas = np.random.randint(0, disponibilidad + 1)
            precio_fijado = np.random.choice(np.arange(40, 101, 5))
            precio_competidor = precio_fijado + np.random.randint(-10, 15)
            ingreso_total = reservas * precio_fijado
            indice_trends = np.random.randint(20, 101)
            datos.append([
                fecha, disponibilidad, reservas, precio_fijado,
                ingreso_total, precio_competidor, indice_trends
            ])
        df = pd.DataFrame(datos, columns=columnas)
        df.to_csv(archivo, index=False)
        print("Archivo historico_alquileres.csv inicializado con datos simulados.")
    else:
        print("El archivo historico_alquileres.csv ya contiene datos.")


if __name__ == "__main__":
    inicializar_historico()


import streamlit as st

# ---------- MODELO Y TENDENCIA GLOBAL ----------
modelo = ModeloPrecios()
indice_trends = modelo.obtener_o_actualizar_tendencia()

st.title("App de Precios Dinámicos para Alquiler de Furgonetas")

# ---------- SECCIÓN 1: DECISIÓN DE PRECIO PARA HOY ----------
st.header("1. Decisión de Precio para Hoy")

st.info(f"Índice Google Trends semanal (más reciente): **{indice_trends}**")

col1, col2 = st.columns(2)
with col1:
    disponibilidad_hoy = st.number_input(
        "Disponibilidad diaria de HOY (número de furgonetas libres)",
        min_value=0, max_value=100, value=5, step=1
    )
with col2:
    precio_competidor_hoy = st.number_input(
        "Precio del competidor HOY (€)",
        min_value=0, max_value=200, value=60, step=1
    )

if st.button("Obtener Precio Propuesto"):
    estado_hoy = modelo._discretizar_estado(disponibilidad_hoy, precio_competidor_hoy, indice_trends)
    precio_optimo = modelo.obtener_precio_optimo(estado_hoy)
    st.header(f"PRECIO ÓPTIMO RECOMENDADO: {precio_optimo} €")


# ---------- SECCIÓN 2: APRENDIZAJE Y ACTUALIZACIÓN DE LA TABLA Q ----------
st.header("2. Aprendizaje y Actualización de la Tabla Q (Resultados de AYER)")
with st.form("form_update_q"):
    st.subheader("Resultados de AYER")
    reservas_ayer = st.number_input("Reservas concretadas AYER", min_value=0, max_value=100, value=2, step=1)
    precio_fijado_ayer = st.number_input("Precio FIJADO AYER (€)", min_value=0, max_value=200, value=60, step=1)
    st.markdown("**Parámetros contextuales de Ayer:**")
    disponibilidad_ayer = st.number_input("Disponibilidad inicial AYER", min_value=0, max_value=100, value=7, step=1)
    precio_competidor_ayer = st.number_input("Precio competidor AYER (€)", min_value=0, max_value=200, value=60, step=1)
    indice_trends_ayer = st.number_input("Índice Google Trends AYER", min_value=0, max_value=100, value=indice_trends, step=1)

    submitted = st.form_submit_button("Actualizar Modelo y Guardar Histórico")

    if submitted:
        recompensa = reservas_ayer * precio_fijado_ayer

        estado_antiguo = modelo._discretizar_estado(disponibilidad_ayer, precio_competidor_ayer, indice_trends_ayer)
        estado_nuevo = modelo._discretizar_estado(disponibilidad_hoy, precio_competidor_hoy, indice_trends)

        try:
            accion_idx = np.where(modelo.precios_posibles == precio_fijado_ayer)[0][0]
        except IndexError:
            accion_idx = 0  # Por si el precio no está en el espacio de acciones

        modelo.actualizar_q_table(estado_antiguo, accion_idx, recompensa, estado_nuevo)

        df_hist = pd.DataFrame([{
            "fecha": datetime.today().strftime("%Y-%m-%d"),
            "disponibilidad_inicial": disponibilidad_ayer,
            "reservas_concretadas": reservas_ayer,
            "precio_fijado": precio_fijado_ayer,
            "ingreso_total": recompensa,
            "precio_competidor": precio_competidor_ayer,
            "indice_google_trends": indice_trends_ayer
        }])

        historico_path = "historico_alquileres.csv"
        if os.path.exists(historico_path):
            df_hist.to_csv(historico_path, mode='a', header=False, index=False)
        else:
            df_hist.to_csv(historico_path, index=False)

        st.success("Modelo Actualizado y Datos Guardados.")


# ---------- SECCIÓN 3: VISUALIZACIÓN DE INGRESOS ----------
st.header("3. Evolución del Ingreso Diario (últimas 30 operaciones)")

if os.path.exists("historico_alquileres.csv"):
    df = pd.read_csv("historico_alquileres.csv")
    if not df.empty:
        df_vis = df.tail(30)
        st.line_chart(
            data=df_vis,
            x="fecha",
            y="ingreso_total",
            use_container_width=True
        )
    else:
        st.info("No hay datos históricos suficientes para mostrar el gráfico.")
else:
    st.info("No existe el archivo de histórico. Añade datos primero.")

