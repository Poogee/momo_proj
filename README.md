# Домашнее задание: ускоренные методы (разделы 1–5)

Решение 5 разделов ДЗ с `hse26.fmin.xyz/homework.html`:

1. **Сопряжённые градиенты** — рандомизированный предобуславливатель Адамара (SRHT) для CG.
2. **Метод Ньютона и квазиньютоновские методы** — зацикливание Ньютона; Hessian-Free Newton для логистической регрессии.
3. **Условные градиентные методы** — Франк–Вульф на многограннике Биркгофа; квадратичная задача на симплексе (FW vs проекционный градиент).
4. **Субградиентный метод** — точка в пересечении выпуклых множеств; LASSO (правила шага + тяжёлый шар).
5. **Проксимальный градиентный метод** — разреженная softmax-регрессия (UCI 697).

Ссылки на факты — конспекты `hse26.fmin.xyz/program.html`, занятия 10–14.

## Содержание

| Файл | Описание |
|---|---|
| `solutions.typ` / `solutions.pdf` | Решение (Typst, доказательства, ссылки на лекции) |
| `notebooks/01..07_*.ipynb` | Исполненные тетрадки (код к задачам) |
| `src_py/*.py` | Исходники тетрадок (percent-формат) |
| `build_nb.py` | Сборка `.ipynb` из `.py` |
| `figures/*.png` | Графики, воспроизводимые из тетрадок |
| `data/students.csv` | UCI 697 (Predicting Students' Dropout) |

## Воспроизведение

```sh
# тетрадки
for s in cg_preconditioned:01 newton_logreg:02 fw_birkhoff:03 fw_simplex:04 \
         subgrad_intersection:05 subgrad_lasso:06 prox_softmax:07; do
  n=${s%%:*}; k=${s##*:}; python3 build_nb.py src_py/$n.py notebooks/${k}_${n}.ipynb
  jupyter nbconvert --to notebook --execute --inplace notebooks/${k}_${n}.ipynb
done
# документ
typst compile solutions.typ solutions.pdf
```

Зависимости: `numpy scipy jax matplotlib scikit-learn pandas`, `typst`, `jupyter`.
