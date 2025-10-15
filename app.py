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
