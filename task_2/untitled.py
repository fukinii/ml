class LogisticRegression:
    
    def __init__(self):
        pass
    
    def fit(
        train_x: T.Tensor,
        train_y: T.Tensor,
        lrn_rate: np.double,
        indices: np.array,
        regular: int = 0
    ) -> T.Tensor:
        
        """Метод для обучения выборки и нахождения коэффициентов w. Оптимизирующий метод - градиентный спуск

        Args:
            epoch: момент времени, когда задана орбита (c)
            semimajor: большая полуось орбиты (м)
            eccentricity: эксцентриситет
            inclination: наклонение (рад)
            argument_periapsis: аргумент перицентра (рад)
            ascending_node: долгота восходящего узла (рад)
            true_anomaly: истинная аномалия (рад)
            central_body: центральное тело, вокруг которого происходит движение
        """

        
        pass