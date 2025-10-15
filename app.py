import numpy as np
import joblib
import os
import pandas as pd
from datetime import datetime, timedelta
from pytrends.request import TrendReq

class ModeloPrecios:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        # Espacio de acciones: precios de 40 a 100 en pasos de 5
        self.espacio_acciones = np.arange(40, 105, 5)
        # Dimensiones del estado
        self.n_disponibilidad = 3
        self.n_competidor = 2
        self.n_tendencia = 2
        # Hiperparámetros
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Inicializa o carga la Tabla Q
        self.q_table = self._cargar_q_table()

    def _cargar_q_table(self):
        if os.path.exists("q_table.pkl"):
            return joblib.load("q_table.pkl")
        shape = (self.n_disponibilidad, self.n_competidor, self.n_tendencia, len(self.espacio_acciones))
        return np.zeros(shape)

    def _guardar_q_table(self):
        joblib.dump(self.q_table, "q_table.pkl")

    def obtener_o_actualizar_tendencia(self):
        """Obtiene y guarda el índice semanal de Google Trends, devuelve el último índice disponible."""
        pytrends = TrendReq(hl='es-ES', tz=360)
        kw_list = ["alquiler de furgonetas"]
        # Último año, granularidad semanal.
        hoy = datetime.today()
        hace_un_ano = hoy - timedelta(days=365)
        timeframe = f"{hace_un_ano.strftime('%Y-%m-%d')} {hoy.strftime('%Y-%m-%d')}"
        pytrends.build_payload(kw_list, cat=478, timeframe=timeframe, geo='ES')
        df_trends = pytrends.interest_over_time()
        if df_trends.empty:
            raise ValueError("No se pudieron obtener datos de Google Trends.")
        # Prepara el dataframe para guardar en CSV
        df_trends = df_trends.reset_index()
        df_cache = pd.DataFrame({
            "fecha_semanal": df_trends["date"].dt.strftime("%Y-%m-%d"),
            "indice_valor": df_trends[kw_list[0]].astype(int)
        })
        df_cache.to_csv("google_trends_cache.csv", index=False)
        # Devuelve el valor más reciente
        return int(df_cache["indice_valor"].iloc[-1])

    def _discretizar_estado(self, disponibilidad, precio_competidor, indice_google_trends):
        # Disponibilidad: 0 si <3, 1 si 3-6, 2 si >6
        if disponibilidad < 3:
            disp = 0
        elif disponibilidad <= 6:
            disp = 1
        else:
            disp = 2
        # Competidor: 0 si <60, 1 si >=60
        comp = 0 if precio_competidor < 60 else 1
        # Tendencia: 0 si <50, 1 si >=50
        tend = 0 if indice_google_trends < 50 else 1
        return (disp, comp, tend)

    def obtener_precio_optimo(self, estado_discretizado):
        """Devuelve el precio óptimo para el estado discreto usando política epsilon-greedy."""
        disp, comp, tend = estado_discretizado
        q_values = self.q_table[disp, comp, tend]
        # Si todos los Q son cero (nunca explorado): explora aleatoriamente con epsilon
        if np.all(q_values == 0) and np.random.random() < self.epsilon:
            accion_idx = np.random.choice(len(self.espacio_acciones))
        else:
            # Con probabilidad epsilon explora, si no, explota
            if np.random.random() < self.epsilon:
                accion_idx = np.random.choice(len(self.espacio_acciones))
            else:
                accion_idx = np.argmax(q_values)
        return self.espacio_acciones[accion_idx]

    def actualizar_q_table(self, estado_antiguo, accion_idx, recompensa, estado_nuevo):
        """Actualiza la tabla Q usando la ecuación de Bellman"""
        disp_old, comp_old, tend_old = estado_antiguo
        disp_new, comp_new, tend_new = estado_nuevo
        q_old = self.q_table[disp_old, comp_old, tend_old, accion_idx]
        q_max_nuevo = np.max(self.q_table[disp_new, comp_new, tend_new])
        # Bellman update
        q_nuevo = (1 - self.alpha) * q_old + self.alpha * (recompensa + self.gamma * q_max_nuevo)
        self.q_table[disp_old, comp_old, tend_old, accion_idx] = q_nuevo
        self._guardar_q_table()


def inicializar_historico():
    archivo = "historico_alquileres.csv"
    # Cabeceras
    columnas = [
        "fecha", "disponibilidad_inicial", "reservas_concretadas",
        "precio_fijado", "ingreso_total", "precio_competidor", "indice_google_trends"
    ]
    # Si el archivo no existe o solo tiene cabecera, lo llenamos
    if not os.path.exists(archivo) or os.stat(archivo).st_size < 60:
        fechas = [ (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(50, 0, -1)]
        datos = []
        for fecha in fechas:
            disponibilidad = np.random.randint(1, 11)  # de 1 a 10
            reservas = np.random.randint(0, disponibilidad+1)
            precio_fijado = np.random.choice(np.arange(40, 105, 5))
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
