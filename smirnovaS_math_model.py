import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.spatial import distance_matrix

# Генерация случайных городов
np.random.seed(42)
n_cities = 5
cities = np.random.rand(n_cities, 2) * 100

# Матрица расстояний
dist_matrix = distance_matrix(cities, cities)

# Полный перебор всех возможных маршрутов
def brute_force_tsp(cities, dist_matrix):
    n = len(cities)
    min_distance = float('inf')
    best_route = None
    
    # Генерируем все возможные перестановки
    for perm in permutations(range(n)):
        # Добавляем возврат в начальный город
        current_distance = 0
        for i in range(n):
            current_distance += dist_matrix[perm[i]][perm[(i+1)%n]]
        
        if current_distance < min_distance:
            min_distance = current_distance
            best_route = perm
    
    return best_route, min_distance

# Решение задачи
best_route, min_distance = brute_force_tsp(cities, dist_matrix)
print(f"Лучший маршрут: {best_route}")
print(f"Минимальное расстояние: {min_distance:.2f}")

# Визуализация
plt.figure(figsize=(10, 6))

# Отображаем города
plt.scatter(cities[:, 0], cities[:, 1], s=200, c='red', marker='o')

# Подписываем города
for i, (x, y) in enumerate(cities):
    plt.text(x, y, str(i), fontsize=12, ha='center', va='center')

# Отображаем лучший маршрут
route = list(best_route) + [best_route[0]]  # Замыкаем маршрут
plt.plot(cities[route, 0], cities[route, 1], 'b-')

plt.title(f'Оптимальный маршрут (расстояние: {min_distance:.2f})')
plt.xlabel('X координата')
plt.ylabel('Y координата')
plt.grid(True)
plt.show()