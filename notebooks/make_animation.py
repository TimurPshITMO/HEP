"""
HEP Evolution Animation
=======================
Создает двухпанельную анимацию эволюции гиперграфа из сохраненной истории.

Панель 1 (слева): Фитнес-кривая, нарастающая кадр за кадром.
                   Моменты "открытия" новых зависимостей подсвечиваются
                   вертикальной пунктирной линией.

Панель 2 (справа): Структура гиперграфа лучшей особи текущего поколения.
                   Использует HyperNetX Euler-bubble визуализацию.

Результат: evolution.gif (в папке notebooks/)
"""

import sys, os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

# --- Настройка путей ---
current_path = os.getcwd()
project_root = os.path.abspath(os.path.join(current_path, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hep_engine.visualizer import HEPVisualizer

# ─────────────────────────────────────────────────────────
# ШАГ 1: Загрузка истории эволюции
# ─────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join('history', 'full_history.json')
OUTPUT_GIF   = 'evolution.gif'
FRAMES_DIR   = os.path.join('history', 'anim_frames')

print("📂 Загрузка истории эволюции...")
with open(HISTORY_FILE, 'r') as f:
    hist_data = json.load(f)
    #labels = hist_data['labels']
    history = hist_data['history']
    n_features = hist_data['n_features']

n_gens = len(history)
print(f"   Найдено поколений: {n_gens}")

# ─────────────────────────────────────────────────────────
# ШАГ 2: Извлечение данных о фитнесе
# ─────────────────────────────────────────────────────────
generations  = [g['generation'] for g in history]
best_fitness = [max(ind['fitness'] for ind in g['population']) for g in history]
avg_fitness  = [np.mean([ind['fitness'] for ind in g['population']]) for g in history]

# Определяем "моменты открытия" — поколения со скачком фитнеса > threshold
JUMP_THRESHOLD = 0.03
discovery_gens = []
for i in range(1, len(best_fitness)):
    if best_fitness[i] - best_fitness[i-1] > JUMP_THRESHOLD:
        discovery_gens.append(i)
print(f"   Обнаружены скачки в поколениях: {[generations[i] for i in discovery_gens]}")

# ─────────────────────────────────────────────────────────
# ШАГ 3: Предварительный рендеринг гиперграфов (PNG-кадры)
# Это самый долгий шаг — каждый граф рисуется один раз и сохраняется.
# Затем анимация просто подгружает готовые картинки через imshow.
# ─────────────────────────────────────────────────────────
os.makedirs(FRAMES_DIR, exist_ok=True)

# Определяем число признаков из истории
#n_features = len(labels)

# Фиксированный круговой layout — одинаковый для всех кадров (иначе граф "прыгает")
import networkx as nx
temp_G = nx.Graph()
temp_G.add_nodes_from(range(n_features))
fixed_pos = nx.circular_layout(temp_G)

viz = HEPVisualizer(output_dir=FRAMES_DIR)

print(f"🎨 Рендеринг {n_gens} кадров гиперграфа...")
frame_paths = []
for i, gen_data in enumerate(history):
    best_ind = max(gen_data['population'], key=lambda x: x['fitness'])
    frame_path = os.path.join(FRAMES_DIR, f"frame_{i:03d}.png")
    
    viz.plot_individual(
        best_ind,
        n_features=n_features,
        title=f"Gen {gen_data['generation']:02d} — Best Genome",
        save_path=frame_path,
        pos=fixed_pos,
        #labels=labels
    )
    frame_paths.append(frame_path)
    if i % 10 == 0:
        print(f"   {i}/{n_gens} кадров готово...")

print("   ✅ Все кадры готовы.")

# ─────────────────────────────────────────────────────────
# ШАГ 4: Сборка двухпанельной анимации через FuncAnimation
# ─────────────────────────────────────────────────────────
print("🎬 Сборка анимации...")

fig = plt.figure(figsize=(18, 8), facecolor='#1a1a2e')
gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)

ax_curve  = fig.add_subplot(gs[0])   # Левая панель — кривая
ax_graph  = fig.add_subplot(gs[1])   # Правая панель — гиперграф

# Стиль левой панели
ax_curve.set_facecolor('#16213e')
ax_curve.set_xlim(-0.5, n_gens - 0.5)
ax_curve.set_ylim(
    min(best_fitness) - 0.05,
    max(best_fitness) + 0.1
)
ax_curve.set_xlabel('Поколение', color='white', fontsize=13)
ax_curve.set_ylabel('Fitness (CV R²)', color='white', fontsize=13)
ax_curve.set_title('Эволюция качества', color='white', fontsize=15, fontweight='bold')
ax_curve.tick_params(colors='white')
for spine in ax_curve.spines.values():
    spine.set_edgecolor('#444466')

# Горизонтальные guideline'ы для ориентира
for val in np.arange(0, 1.0, 0.1):
    ax_curve.axhline(val, color='#333355', lw=0.5, ls='--')

# Правая панель — тёмный фон
ax_graph.set_facecolor('#1a1a2e')
ax_graph.set_axis_off()

# Статичные объекты (обновляются внутри animate)
line_best, = ax_curve.plot([], [], color='#00e5ff', lw=2.5, label='Best')
line_avg,  = ax_curve.plot([], [], color='#ff9800', lw=1.5, ls='--', alpha=0.7, label='Average')
gen_text   = ax_curve.text(0.02, 0.97, '', transform=ax_curve.transAxes,
                            color='white', fontsize=11, va='top')
ax_curve.legend(loc='lower right', facecolor='#16213e', labelcolor='white', fontsize=10)

# Вертикальные маркеры "прорыва" (рисуем статично — они всегда видны)
for di in discovery_gens:
    ax_curve.axvline(x=generations[di], color='#ff4081', lw=1.5, ls=':', alpha=0.8)
    ax_curve.text(generations[di] + 0.3, min(best_fitness),
                  '⚡', color='#ff4081', fontsize=12)

# Заглушка изображения для правой панели
dummy_img = np.ones((100, 100, 3))
img_display = ax_graph.imshow(dummy_img, aspect='auto')

def animate(frame_idx):
    # --- Левая панель: кривая нарастает ---
    x_data = generations[:frame_idx + 1]
    line_best.set_data(x_data, best_fitness[:frame_idx + 1])
    line_avg.set_data(x_data, avg_fitness[:frame_idx + 1])
    gen_text.set_text(
        f"Gen: {generations[frame_idx]:02d}  |  "
        f"Best: {best_fitness[frame_idx]:.4f}"
    )

    # --- Правая панель: подгружаем готовый PNG кадра ---
    img = np.array(Image.open(frame_paths[frame_idx]).convert('RGB'))
    img_display.set_data(img)
    img_display.set_extent([0, img.shape[1], img.shape[0], 0])
    ax_graph.set_xlim(0, img.shape[1])
    ax_graph.set_ylim(img.shape[0], 0)

    return line_best, line_avg, gen_text, img_display

anim = FuncAnimation(
    fig,
    animate,
    frames=n_gens,
    interval=300,      # 300ms на кадр → ~3 fps (плавно, но видно каждое поколение)
    blit=False
)

# ─────────────────────────────────────────────────────────
# ШАГ 5: Сохранение GIF
# ─────────────────────────────────────────────────────────
print(f"💾 Сохранение в '{OUTPUT_GIF}'...")
writer = PillowWriter(fps=3)
anim.save(OUTPUT_GIF, writer=writer, dpi=100)
plt.close()

print(f"\n✅ Готово! Анимация сохранена: {os.path.abspath(OUTPUT_GIF)}")
print(f"   Размер файла: {os.path.getsize(OUTPUT_GIF) / 1024:.1f} KB")
