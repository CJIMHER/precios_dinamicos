import numpy as np
import joblib
import os

class ModeloPrecios:
    def __init__(self):
        # Espacio de acciones: precios de 40 a 100 en pasos de 5
        self.espacio_acciones = np.arange(40, 105, 5)

        # Dimensiones del estado
        self.n_disponibilidad = 3  # 3 niveles de disponibilidad
        self.n_competidor = 2      # 2 niveles de precio competidor
        self.n_tendencia = 2       # 2 niveles de tendencia Google Trends

        # Inicializa o carga la Tabla Q
        self.q_table = self._cargar_q_table()

    def _cargar_q_table(self):
        """Carga la tabla Q desde q_table.pkl o la inicializa si no existe."""
        if os.path.exists("q_table.pkl"):
            return joblib.load("q_table.pkl")
        # Inicializa la Q-table como ceros
        shape = (self.n_disponibilidad, self.n_competidor, self.n_tendencia, len(self.espacio_acciones))
        return np.zeros(shape)

    def _guardar_q_table(self):
        """Guarda la tabla Q en q_table.pkl."""
        joblib.dump(self.q_table, "q_table.pkl")

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

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

# Llama a la función cuando se ejecute el script
if __name__ == "__main__":
    inicializar_historico()
    
