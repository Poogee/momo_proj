#set page(paper: "a4", margin: (x: 1.8cm, y: 2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "ru")
#set par(justify: true)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => [#pagebreak(weak: true) #block(it)]
#set math.equation(numbering: "(1)")

#let nm(x) = $lr(bar.v.double #x bar.v.double)$
#let ip(a, b) = $lr(chevron.l #a, #b chevron.r)$
#let argmin = math.op("arg min", limits: true)
#let prox = math.op("prox")
#let dist = math.op("dist")
#let proj = math.op("proj")
#let R = math.bb("R")
#let T = math.sans("T")
#let lec(n) = box(fill: luma(235), inset: (x: 3pt, y: 1pt), radius: 2pt)[#text(size: 8.5pt)[Лекция #n]]

#align(center)[
  #text(17pt, weight: "bold")[Домашнее задание: ускоренные методы — разделы 1–5]
  #v(2pt)
  #text(10pt)[Сопряжённые градиенты · Ньютон/квази-Ньютон · Условный градиент · Субградиентный · Проксимальный]
  #v(2pt)
  #text(9pt, style: "italic")[Источник: `hse26.fmin.xyz/homework.html`. Все ссылки — на конспекты `hse26.fmin.xyz/program.html` (занятия 10–14). Код: каталог `notebooks/`, рисунки воспроизводимы.]
]

#outline(depth: 2, indent: auto)

= Сопряжённые градиенты

== Рандомизированные предобуславливатели для CG (20)

Задача наименьших квадратов $f(x) = 1/2 nm(hat(A) x - hat(b))_2^2$, $hat(A) in R^(m times n)$, $m gt.double n$.
Минимум $arrow.r.double.long$ нормальные уравнения
$ A x = b, quad A := hat(A)^top hat(A) in R^(n times n), quad b := hat(A)^top hat(b) in R^n. $

#lec(10) Теорема 2 (сходимость CG как итерационного метода):
$ nm(x_k - x^*)_A <= 2 lr(( (sqrt(kappa(A)) - 1)/(sqrt(kappa(A)) + 1) ))^k nm(x_0 - x^*)_A, quad kappa(A) = lambda_1(A)\/lambda_n(A). $
Поскольку $kappa(hat(A)^top hat(A)) = kappa^2(hat(A))$, обусловленность возводится в квадрат, поэтому нужен предобуславливатель: переход $A x = b arrow.r M A x = M b$ при $kappa(M A) lt.double kappa(A)$; спектр $M A$ совпадает со спектром $T^top A T$ #lec(10). Идеал $M = A^(-1)$ недостижим, ищем дешёвый $M approx A^(-1)$.

Сэмплирующая матрица $Phi = R H_m^("norm") S in R^((n+p) times m)$ ($S = "diag"(plus.minus 1)$, $R$ — выбор строк, $H_m^("norm") = m^(-1/2) H_m$), и
$ M^(-1) = hat(A)^top Phi^top Phi hat(A) = (Phi hat(A))^top (Phi hat(A)), quad W := Phi hat(A) in R^((n+p) times n). $

=== (2) FLOP-ы для $M^(-1)$ и $M$
Строим $W = R H_m^("norm") S hat(A)$ по столбцам ($n$ столбцов):
- $S hat(A)$: домножение на диагональ — $O(m n)$;
- $H_m^("norm") (S hat(A))$: быстрое преобразование Адамара (matvec $H_m v$ за $m log m$) к каждому из $n$ столбцов плюс масштаб $m^(-1/2)$ — $O(n m log m)$;
- $R(dot)$: выбор $n+p$ строк — $O((n+p) n)$ копирований (без умножений).

Итого $W$: $O(n m log m)$. Далее $M^(-1) = W^top W$: $(n+p) times n$ на $n times (n+p)$ — $O(n^2 (n+p))$. Обращение $M = (M^(-1))^(-1)$ (или фактор Холецкого $M^(-1) = L L^top$): $O(n^3)$.
$ #box(stroke: 0.6pt, inset: 5pt)[$ "FLOPs"(M^(-1)) = O(n m log m + n^2(n+p)), quad "FLOPs"(M) = O(n^3). $] $

=== (2) FLOP-ы наивного $hat(A)^top hat(A)$
$(hat(A)^top hat(A))_(i j) = sum_(k=1)^m hat(A)_(k i) hat(A)_(k j)$ — $n^2$ скалярных произведений длины $m$, симметрия даёт фактор $1/2$:
$ "FLOPs"(hat(A)^top hat(A)) = n^2 (2 m - 1) approx 2 m n^2 = O(m n^2). $

=== (2) FLOP-ы $hat(A)^top hat(A) v$ через $u = hat(A) v$
$u = hat(A) v$: $m times n$ на вектор — $approx 2 m n$. Затем $hat(A)^top u$: $approx 2 m n$.
$ "FLOPs"(hat(A)^top (hat(A) v)) approx 4 m n = O(m n) lt.double O(m n^2). $
Поэтому в CG матрица $A$ никогда не формируется явно.

=== (4) PCG за $k$ итераций против прямого решения
Предобработка PCG: построение $M$ — $O(n m log m + n^3)$ (п. 1.1.1).
Одна итерация PCG: один $A p = hat(A)^top(hat(A) p)$ — $O(m n)$; применение предобуславливателя $z = M r$ через факторы Холецкого $M^(-1)=L L^top$ — $O(n^2)$; векторные операции $O(n)$. Итого
$ T_("PCG")(k) = O(underbrace(n m log m + n^3, "setup") + k(m n + n^2)). $
Прямое решение $A x = b$: формирование $hat(A)^top hat(A)$ — $O(m n^2)$; Холецкий — $O(n^3\/3)$; обратная подстановка — $O(n^2)$:
$ T_("direct") = O(m n^2 + n^3). $
При $log m lt.double n$ (здесь $log_2 4096 = 12 lt.double 400$) setup PCG $O(n m log m) lt.double O(m n^2)$, дешевле формирования $hat(A)^top hat(A)$. Сравнивая ведущие члены $k dot m n$ и $m n^2$:
$ #box(stroke: 0.6pt, inset: 5pt)[$ k dot m n gt.tilde m n^2 quad arrow.r.double.long quad k gt.tilde n. $] $
CG медленнее прямого метода при $k = O(n)$ (здесь $n = 400$). В точной арифметике CG сходится за $<= n$ шагов (#lec(10) Теорема 1: за $r$ — число различных собственных значений), смысл предобуславливания — добиться $k lt.double n$.

=== (10) Реализация (`notebooks/01_cg_preconditioned.ipynb`)
$m = 2^12 = 4096$, $n = 400$, $p = 20$; столбцы $hat(A)$ масштабированы логарифмически $arrow.r.double$ большое $kappa$. Быстрое Адамара проверено против рекурсивного определения $H_m$ (порядок 8). Предобусловленный CG (#lec(10), алгоритм):
$ r_0 = b - A x_0, quad z_0 = M r_0, quad d_0 = z_0; quad
  alpha_k = (r_k^top z_k)/(d_k^top A d_k), quad
  beta_k = (r_(k+1)^top z_(k+1))/(r_k^top z_k). $
Полученные числа:
$ kappa(hat(A)^top hat(A)) approx 1.20 dot 10^6, quad
  kappa(M^(1/2) hat(A)^top hat(A) M^(1/2)) approx 1.45 dot 10^5 quad (approx 8.2 times "меньше"). $
Невязка после 400 итераций: $7.0 dot 10^(-5)$ (без $M$) против $2.8 dot 10^(-6)$ (с $M$). При предписанном малом $p=20$ эскиз SRHT лишь слабо переопределён ($n+p = 420 < n log n$), поэтому спектр кластеризуется умеренно — но CG с предобуславливателем всё равно сходится заметно быстрее.

#figure(image("figures/cg_residual.png", width: 78%),
  caption: [Относительная невязка $nm(hat(A)^top hat(b) - hat(A)^top hat(A) x_k)_2 \/ nm(hat(A)^top hat(b))_2$.])

= Метод Ньютона и квазиньютоновские методы

== Проблема сходимости Ньютона (10)

$ f(x,y) = x^4/4 - x^2 + 2 x + (y-1)^2, quad
  nabla f = vec(x^3 - 2 x + 2, 2(y-1)), quad
  nabla^2 f = mat(3 x^2 - 2, 0; 0, 2). $

Переменные разделяются. По $y$: $f$ квадратична, Ньютон попадает в $y^* = 1$ за один шаг при любом старте.

По $x$: $g(x) = x^4/4 - x^2 + 2 x$, $g'(x) = x^3 - 2 x + 2$, $g''(x) = 3 x^2 - 2$. Единственный вещественный корень $g'$: $x^* approx -1.7693$, $g''(x^*) approx 7.39 > 0$ — глобальный минимум ($g$ коэрцитивна, $x^4$ доминирует).

*Итерации Ньютона из $x_0 = 0$.* $x_(k+1) = x_k - g'(x_k)\/g''(x_k)$:
$ x_0 = 0: && g'(0)=2, g''(0)=-2 quad arrow.r quad x_1 = 0 - 2/(-2) = 1, \
  x_1 = 1: && g'(1)=1, g''(1)=1 quad arrow.r quad x_2 = 1 - 1/1 = 0, quad arrow.r.double quad {x_k} = {0,1,0,1,dots}. $

#box(stroke: 0.6pt, inset: 5pt)[Метод Ньютона *зацикливается* в $2$-цикле $\{0, 1\}$ и не сходится к $x^* approx -1.77$.]

*Объяснение* #lec(11): локальная квадратичная сходимость гарантируется лишь при $mu I_n prec.eq nabla^2 f prec.eq L I_n$ и хорошей инициализации $nm(x_0 - x^*) < 2 mu \/ M$ ($M$ — константа Липшица гессиана). Здесь в $x_0=0$ гессиан $g''(0) = -2 < 0$ — *индефинитен*, направление Ньютона не является направлением спуска, предположение $nabla^2 f succ.eq mu I$ нарушено (ср. #lec(11) «Отсутствие квадратичной сходимости, если предположения нарушаются», «Проблемы метода Ньютона»).

*Градиентный спуск $alpha = 0.01$.* $x_(k+1) = x_k - 0.01 (x_k^3 - 2 x_k + 2)$. Малый шаг $arrow.r.double$ монотонное убывание $f$ (т.к. $alpha < 2\/L$ локально); $f$ коэрцитивна и имеет единственный минимум $arrow.r.double$ сходимость к $(x^*, 1) approx (-1.7693, 1)$, медленно (линейно).

*Наискорейший спуск* (точный линейный поиск, #lec(10)). Тоже метод спуска: $f(x_(k+1)) <= f(x_k)$, на коэрцитивной функции с единственным минимумом сходится к $(x^*, 1)$; направления соседних шагов ортогональны (#lec(10)), на овражной части возможен зигзаг.

== Hessian-Free метод Ньютона (20)

Бинарная логистическая регрессия, $p(y=1|x;w) = sigma(w^top x)$, $sigma(t)=1/(1+e^(-t))$:
$ f(w) = sum_(i=1)^m [ log(1 + e^(w^top x_i)) - y_i w^top x_i ] + mu/2 nm(w)_2^2. $
$ nabla f(w) = X^top (sigma(X w) - y) + mu w, quad
  nabla^2 f(w) = X^top D X + mu I, quad D = "diag"(sigma_i(1 - sigma_i)). $
Так как $0 < sigma(1-sigma) <= 1/4$:
$ mu I prec.eq nabla^2 f(w) prec.eq L I, quad L = 1/4 lambda_max(X^top X) + mu, $
т.е. $f$ есть $L$-гладкая и (при $mu>0$) $mu$-сильно выпуклая (#lec(11), $L$-гладкость и сильная выпуклость; $kappa = L\/mu$). Данные: $m=1000$, $d=100$, $y_i = bb(1)[sigma(x_i^top w^*) > 0.5]$ $arrow.r.double$ *линейно разделимы*. Эксперименты — `notebooks/02_newton_logreg.ipynb`; $lambda_max(X^top X) approx 1.70 dot 10^3$.

*(a) GD, $mu=1$.* $L approx 426.2$, теоретический максимальный шаг $2\/L approx 4.69 dot 10^(-3)$. Шаги $0.5\/L, 1\/L, 1.9\/L$ дают линейную сходимость (сильная выпуклость), шаг $2.05\/L$ расходится — граница $2\/L$ точна.

*(b) Ньютон, $mu=1$.* Все предположения #lec(11) выполнены ($mu I prec.eq nabla^2 f prec.eq L I$, гессиан липшицев) $arrow.r.double$ локальная квадратичная сходимость $nm(w_(k+1)-w^*) <= (M\/2mu) nm(w_k - w^*)^2$; численно $f - f^* < 10^(-3)$ за $approx 6$ итераций ($f^* = 97.3743$).

*(c) Демпфированный Ньютон.* Шаг $w_(k+1) = w_k - t_k [nabla^2 f]^(-1) nabla f$ с backtracking (условие Армихо). Глобальная сходимость из любой точки + переход к $t_k = 1$ и квадратичной фазе вблизи $w^*$ — устраняет «дикие» шаги чистого Ньютона.

*(d) GD, $mu=0$.* Разделимость $arrow.r.double$ $inf f = 0$ *не достигается* ($nm(w_k) -> infinity$); нет сильной выпуклости $arrow.r.double$ только $O(1\/k)$. $L_0 = 1/4 lambda_max(X^top X)$; шаг $1\/L_0$ убывает $f$ сублинейно (численно $f$ застревает $approx 35.5$ за 400 итераций). $epsilon$-точность по $f$ достижима (т.к. $f^*=0$), но по $nm(nabla f)$ — крайне медленно (минимум на бесконечности).

*(e) Ньютон, $mu=0$.* При $w$ с уверенными предсказаниями $D = sigma(1-sigma) -> 0$ $arrow.r.double$ $nabla^2 f = X^top D X$ почти вырожден, нарушено $mu I prec.eq nabla^2 f$ (#lec(11)) $arrow.r.double$ огромные шаги, нет квадратичной сходимости. Демпфированный вариант остаётся устойчив: численно $f -> 8.6 dot 10^(-10)$ (приближение к $0$).

*(f) Ньютон–CG.* $nabla^2 f(w_k) d_k = -nabla f(w_k)$ решается CG (`jax.scipy.sparse.linalg.cg`, #lec(10)); затем $w_(k+1) = w_k + alpha d_k$. Тот же квадратичный профиль, что у точного Ньютона, при неточном (CG) решении системы.

*(g) Hessian-Free Newton.* Произведение гессиан-вектор через автодифф $nabla^2 f(w) v = nabla_w [ ip(nabla f(w), v) ]$ (forward-over-reverse, `jax.jvp(jax.grad f)`), гессиан не хранится. Память $O(d)$ против $O(d^2)$. Замеры (15 итераций, $mu=1$, $f - f^*$): явный Ньютон $approx 141$ мс, Ньютон–CG (плотный $H$) $approx 386$ мс, HFN $approx 561$ мс. При малом $d=100$ плотный гессиан ($d^2 = 10^4$ чисел) дешевле; преимущество HFN — память $O(d)$ и масштабируемость на большие $d$, где формирование $nabla^2 f$ невозможно.

#grid(columns: 2, gutter: 6pt,
  figure(image("figures/newton_gd_mu1.png"), caption: [(a) GD, $mu=1$: граница $2\/L$.]),
  figure(image("figures/newton_mu1.png"), caption: [(b,c) Ньютон vs демпфированный, $mu=1$.]),
  figure(image("figures/newton_mu0.png"), caption: [(d,e) $mu=0$: $inf f = 0$ недостижим.]),
  figure(image("figures/newton_hfn.png"), caption: [(f,g) Ньютон / Ньютон–CG / HFN.]))

= Условные градиентные методы

== Проекция на многогранник Биркгофа методом Франк–Вульфа (20)

$min_(X in B_n) f(X) = 1/2 nm(X - Y)_F^2$, $B_n = \{ X >= 0, X bb(1) = bb(1), X^top bb(1) = bb(1) \}$.

=== (5) Градиент и оракул LMO
$ nabla f(X) = X - Y. $
Шаг LMO (#lec(12), идея Франк–Вульфа: $y_k = argmin_(x in S) ip(nabla f(x_k), x)$, $x_(k+1) = gamma_k x_k + (1-gamma_k) y_k$):
$ S_k = argmin_(S in B_n) ip(nabla f(X_k), S) = argmin_(S in B_n) ip(X_k - Y, S). $
Линейная функция на многограннике достигает минимума в вершине. *Теорема Биркгофа–фон Неймана*: вершины $B_n$ — в точности матрицы перестановок. Поэтому
$ min_(S in B_n) ip(G, S) = min_(pi in S_n) sum_(i=1)^n G_(i, pi(i)) $
— *линейная задача о назначениях*; решается венгерским алгоритмом (`scipy.optimize.linear_sum_assignment`). $S_k$ — *матрица перестановки*.

=== (10–5) Реализация (`notebooks/03_fw_birkhoff.ipynb`)
Точный линейный поиск для квадратичной $f$: $phi(gamma) = f(X_k + gamma(S_k - X_k))$, $phi'(gamma) = ip(X_k - Y + gamma(S_k - X_k), S_k - X_k) = 0$ $arrow.r.double$
$ gamma_k = (ip(X_k - Y, X_k - S_k))/(nm(X_k - S_k)_F^2) "  (отсечение в " [0,1] ") — совпадает с заданным." $
Старт $X_0 = I$ (перестановка). Итерации — выпуклые комбинации матриц перестановок $arrow.r.double$ $X_k in B_n$ *точно* на каждом шаге: численно $max|sum_j X_(i j) - 1| approx 9 dot 10^(-16)$, $max|sum_i X_(i j) - 1| approx 7 dot 10^(-16)$, отрицательных элементов нет; $f(X_200) approx 6.413$. Сходимость $O(1\/k)$ (#lec(12), $f(x_k) - f^* <= 2 L R^2 \/ (k+1)$).

#figure(image("figures/fw_birkhoff.png", width: 62%),
  caption: [Франк–Вульф на $B_5$, LMO = венгерский алгоритм, скорость $O(1\/k)$.])

== Минимизация квадратичной функции на симплексе (20)

$min_(x in Delta_n) f(x) = 1/2 x^top Q x + c^top x$, $Q in bb(S)_(+ +)^n$, $nabla f(x) = Q x + c$.

=== (5) Данные
$n=20$; $Q = U "diag"(s) U^top$ со спектром $s subset [mu, L] = [1, 100]$ ($U$ ортогональна); $x^* tilde "Dirichlet"(bb(1)) in "ri"(Delta_n)$; $c = -Q x^*$. Тогда безусловный минимум $-Q^(-1)c = x^*$ лежит в $Delta_n$ $arrow.r.double$ $x^*$ — и условный минимум, $f^* = f(x^*)$. Старты: вершина $e_1$ и барицентр $bb(1)\/n$.

=== (7) Франк–Вульф
LMO на симплексе: $min_(s in Delta_n) ip(g, s) = e_(i^*)$, $i^* = argmin_i g_i$ (линейная функция на $Delta_n$ — в вершине). Точный линейный поиск (квадратичная $f$, направление $d = s - x$):
$ gamma_k = "clip"( - ip(g, d) / (d^top Q d), 0, 1 ). $

=== (8) Проекционный градиентный спуск
$ x_(k+1) = proj_(Delta_n)(x_k - 1/L nabla f(x_k)), quad L = lambda_max(Q). $
*Проекция на симплекс* (Duchi и др., 2008) — решение $min_z 1/2 nm(z - v)_2^2$ при $bb(1)^top z = 1$, $z >= 0$. KKT: $z_i = (v_i - theta)_+$, $theta$ — множитель условия $sum z_i = 1$. Сортируем $u = "sort"_(arrow.b)(v)$, $rho = max\{ j : u_j - (1\/j)(sum_(r<=j) u_r - 1) > 0 \}$, $theta = (1\/rho)(sum_(r<=rho) u_r - 1)$, $z = (v - theta)_+$. Оператор проекции нерастягивающий (неравенство Бурбаки–Чейни–Гольдштейна, #lec(12)). Шаг $1\/L$ обоснован #lec(12): для гладкого выпуклого $f(x_k)-f^* <= L nm(x_0-x^*)^2\/(2k)$; для $mu$-сильно выпуклого ($Q succ.eq mu I$) — линейная скорость.

=== Сравнение (`notebooks/04_fw_simplex.ipynb`)
$kappa(Q) = L\/mu = 100$. Численно: PGD сходится *линейно* (зазор $approx 10^(-12)$ при старте $e_1$, $approx 10^(-10)$ из барицентра) — сильная выпуклость. FW *сублинейно* $O(1\/k)$ (зазор $approx 2.4 dot 10^(-3)$): оптимум $x^* in "ri"(Delta_n)$ не вершина, FW зигзагует — это проявление нижней оценки $min(n\/2, L R^2 \/ 16 epsilon)$ для методов с оракулом LMO (#lec(12)).

#figure(image("figures/fw_simplex.png", width: 92%),
  caption: [Квадратичная задача на $Delta_20$: FW $O(1\/k)$ против линейного PGD.])

= Субградиентный метод

== Точка в пересечении выпуклых множеств (30)

$U = \{ nm(A(x-y))_2 <= 1 \}$ (эллипсоид), $V = \{ nm(Sigma x)_oo <= 1 \}$ (брус).

=== (10) Постановка и алгоритм
$ phi.alt(x) = max \{ dist(x, U), dist(x, V) \}, quad dist(x, C) = min_(z in C) nm(x - z)_2. $
$dist(dot, C)$ выпукла (#lec(12), проекция на выпуклое корректна; $1/2 dist^2(x,C)$ дифференцируема с градиентом $x - proj_C(x)$). Максимум выпуклых — выпуклая (#lec(13)). $phi.alt >= 0$ и
$ phi.alt(x) = 0 quad <=> quad x in U inter V. $
Субградиент $dist(dot,C)$ при $x in.not C$: $nabla dist(x,C) = (x - proj_C(x)) \/ nm(x - proj_C(x))$ (из критерия проекции #lec(12)). Субдифференциал максимума — теорема Дубовицкого–Милютина (#lec(13)):
$ partial phi.alt(x) = "conv" union.big_(i in I(x)) partial f_i(x), quad I(x) = \{ i : f_i(x) = phi.alt(x) \}, $
берём субградиент *активной* (большей) дистанции. Так как $phi.alt^* = 0$ известно — *шаг Полякa*
$ x_(k+1) = x_k - (phi.alt(x_k) - phi.alt^*)/(nm(g_k)_2^2) g_k, quad g_k in partial phi.alt(x_k), $
сходится к $U inter V$ (#lec(13), базовое неравенство и сходимость при $sum alpha_k = infinity$, $sum alpha_k^2 < infinity$; Поляк удовлетворяет условиям).

*Проекции.* $proj_V(x)_i = "clip"(x_i, -1\/sigma_i, 1\/sigma_i)$ (брус, покоординатно). $proj_U$: при $nm(A(x-y)) <= 1$ — тождество; иначе KKT для $min_w 1/2 nm(w-c)^2$, $nm(A w)^2 <= 1$ ($w = z - y$, $c = x - y$): $w = (I + lambda A^top A)^(-1) c$, $lambda >= 0$ из $nm(A w)^2 = 1$; в собственном базисе $A^top A = Q Lambda Q^top$ функция $sum_i lambda_i tilde(c)_i^2 \/ (1 + lambda lambda_i)^2$ монотонна по $lambda$ $arrow.r.double$ бисекция.

=== (15) Реализация (`notebooks/05_subgrad_intersection.ipynb`)
$n=2$, $y=(3,2)$, $sigma=(0.5,1)$, $A = mat(1,0;-1,1)$. Старт $x_0=(1,2)$ сходится в $(2,1)$ с $phi.alt = 0$ *точно* и быстрее всех (точка уже почти в одном множестве, активная дистанция мгновенно зануляется). Старты $(2,-1)$ и $(0,0)$ медленнее: $x -> (2, 0.97)$, $phi.alt -> 4.4 dot 10^(-4)$ за 2000 итераций.

=== (5) Обсуждение
- *Выпуклость*: $phi.alt$ выпукла (max выпуклых дистанций), задача выпукла.
- *Гладкость*: $phi.alt$ *негладкая* — изломы там, где $dist(x,U)=dist(x,V)$, и в точках границы (где $dist$ не дифференцируема). Скорость $O(1\/sqrt(k))$ (#lec(13)).
- *Единственность*: решение *не единственно* — всё множество $U inter V$ оптимально.
- *Скорость*: зависит от геометрии и старта; ближайший к $U inter V$ / лежащий внутри одного множества старт сходится быстрее, старт «по диагонали» от тонкого пересечения — медленнее.

#figure(image("figures/subgrad_intersection.png", width: 72%),
  caption: [Субградиентный метод (шаг Полякa), $phi.alt^* = 0$; разные старты.])

== Субградиентные методы для LASSO (10)

$ f(x) = 1/2 nm(A x - b)_2^2 + lambda nm(x)_1. $

*Субдифференциал* (теорема Моро–Рокафеллара, #lec(13): $partial(sum f_i) = sum partial f_i$, $partial(g(A x+b))=A^top partial g$):
$ partial f(x) = A^top (A x - b) + lambda thin partial nm(x)_1, quad
  [partial nm(x)_1]_i = cases(
    {"sign"(x_i)}\, & x_i != 0,
    [-1, 1]\, & x_i = 0). $
Канонический субградиент $g = A^top(A x - b) + lambda "sign"(x)$.

*Стратегии шага и оценки* (#lec(13), базовое неравенство $f_k^("best") - f^* <= (R^2 + G^2 sum alpha_k^2)/(2 sum alpha_k)$):
$ "пост.: " (R^2)/(2 alpha k) + (alpha G^2)/2; quad
  "пост. длины: " (G R^2)/(2 gamma k) + (G gamma)/2; quad
  alpha_k = R/(G sqrt(k+1)): (G R(2 + ln k))/(4 sqrt(k+1)); quad
  alpha_k tilde 1/k; quad
  "Поляк: " alpha_k = (f(x_k)-f^*)/(nm(g_k)^2). $

*Реализация* (`notebooks/06_subgrad_lasso.ipynb`): $n=1000$, $m=200$, $lambda=0.01$, разреженный $x^*$ (20 ненулей). Эталон $f^* = 0.18252$ получен FISTA ($2 dot 10^4$ итераций, #lec(14)). Среди правил шага Поляк сходится быстрее всех; правила $1\/k$ и постоянный шаг — медленно ($O(1\/sqrt(k))$). *Тяжёлый шар* $x_(k+1) = x_k - alpha g_k + beta(x_k - x_(k-1))$: настройка $beta$ даёт зазор $f_k^("best") - f^*$: $0.30 (beta=0)$, $0.25 (0.5)$, $0.17 (0.8)$, $bold(2.9 dot 10^(-2)) (beta = 0.95)$ — импульс существенно ускоряет.

#figure(image("figures/subgrad_lasso.png", width: 95%),
  caption: [LASSO: правила выбора шага (слева) и тяжёлый шар (справа).])

= Проксимальный градиентный метод

== Проксимальный метод для разреженной softmax-регрессии (20)

$ min_(W in R^(c times d)) underbrace(- sum_(i=1)^N log P(y_i | x_i; W), f(W)) + lambda nm(W)_1, quad
  P(y=j|x;W) = e^(W_j^top x) \/ sum_(l=1)^c e^(W_l^top x). $
Датасет «Predicting Students' Dropout and Academic Success» (UCI 697), 3 класса {Dropout, Enrolled, Graduate}, $N=4424$, $d=37$ (с bias), стандартизация. Эксперименты: `notebooks/07_prox_softmax.ipynb`.

=== (4) Точные правила обновления
$ nabla f(W) = (P - Y)^top X, quad P_(i j) = "softmax"(X W^top)_(i j), quad Y "— one-hot." $
*Субградиентный метод:*
$ W_(k+1) = W_k - alpha thin ( nabla f(W_k) + lambda thin "sign"(W_k) ). $
*Проксимальный градиентный (ISTA)* — из условий оптимальности $0 in nabla f(W^*) + partial r(W^*)$ (#lec(14)):
$ W^* = prox_(alpha r)(W^* - alpha nabla f(W^*)) quad arrow.r.double quad
  W_(k+1) = prox_(alpha lambda nm(dot)_1)( W_k - alpha nabla f(W_k) ). $
Прокс $ell_1$ — *мягкий порог* (#lec(14), вывод по трём случаям $x_i>0$, $x_i<0$, $x_i=0$ из $0 in partial h(x_i)$):
$ [prox_(tau nm(dot)_1)(v)]_i = "sign"(v_i) [ abs(v_i) - tau ]_+, quad tau = alpha lambda. $
Итого $W_(k+1) = cal(S)_(alpha lambda)(W_k - alpha nabla f(W_k))$. Скорость #lec(14): $phi.alt(W_k) - phi.alt^* <= L nm(W_0 - W^*)^2 \/ (2 k) = O(1\/k)$ при $alpha = 1\/L$; ускоренный (FISTA) — $O(1\/k^2)$. Критерий оптимальности — градиентное отображение $G_alpha(W) = 1/alpha(W - prox_(alpha r)(W - alpha nabla f(W)))$, $G_alpha = 0 <=> W$ оптимальна (#lec(14)).

Граница гладкости softmax-кросс-энтропии: $nabla^2 f prec.eq 1/2 X^top X arrow.r.double L = 1/2 lambda_max(X^top X) approx 1.16 dot 10^4$, шаг $1\/L approx 8.6 dot 10^(-5)$.

=== (6) $lambda = 0$
Прокс с $tau=0$ — тождество $arrow.r.double$ ISTA $equiv$ GD; максимальный шаг $< 2\/L$ (#lec(11), $L$-гладкость). Численно: NLL $-> 1951.39$, тест-точность $0.765$, разреженность $0\%$ (нет $ell_1$). Субградиент с $alpha_k tilde 1\/sqrt(k)$ медленнее ($O(1\/sqrt(k))$ против $O(1\/k)$, #lec(13)–#lec(14)).

#figure(image("figures/prox_lambda0.png", width: 70%),
  caption: [$lambda=0$: проксимальный (= GD, $O(1\/k)$) против субградиента ($O(1\/sqrt(k))$).])

=== (10) Таблица сходимости
Чистый ISTA, шаг $1\/L$; критерий — относительный зазор $(phi.alt(W_k) - phi.alt^*)\/(phi.alt(W_0) - phi.alt^*) <= epsilon$, эталон $phi.alt^*$ на каждое $lambda$ — FISTA ($4 dot 10^4$ итераций, #lec(14)).

#align(center)[#table(
  columns: 5, align: center, inset: 4pt, stroke: 0.4pt,
  table.header([$lambda$], [$epsilon$], [итераций], [разреженность], [тест-точность]),
  [$10^(-3)$],[$10^(-1)$],[30],[2.70%],[0.7582],
  [$10^(-3)$],[$10^(-2)$],[310],[2.70%],[0.7672],
  [$10^(-3)$],[$10^(-3)$],[970],[2.70%],[0.7627],
  [$10^(-3)$],[$10^(-4)$],[2180],[2.70%],[0.7661],
  [$10^(-3)$],[$10^(-5)$],[3910],[2.70%],[0.7650],
  [$10^(-2)$],[$10^(-1)$],[30],[2.70%],[0.7582],
  [$10^(-2)$],[$10^(-2)$],[310],[2.70%],[0.7672],
  [$10^(-2)$],[$10^(-3)$],[970],[2.70%],[0.7627],
  [$10^(-2)$],[$10^(-4)$],[2250],[2.70%],[0.7661],
  [$10^(-2)$],[$10^(-5)$],[38930],[10.81%],[0.7650],
  [$10^(-1)$],[$10^(-1)$],[30],[2.70%],[0.7582],
  [$10^(-1)$],[$10^(-2)$],[310],[3.60%],[0.7672],
  [$10^(-1)$],[$10^(-3)$],[1000],[3.60%],[0.7638],
  [$10^(-1)$],[$10^(-4)$],[4760],[13.51%],[0.7661],
  [$10^(-1)$],[$10^(-5)$],[139350],[34.23%],[0.7650],
  [$1$],[$10^(-1)$],[30],[4.50%],[0.7582],
  [$1$],[$10^(-2)$],[300],[9.01%],[0.7638],
  [$1$],[$10^(-3)$],[1330],[21.62%],[0.7638],
  [$1$],[$10^(-4)$],[13250],[36.04%],[0.7627],
  [$1$],[$10^(-5)$],[19260],[36.04%],[0.7627],
)]

Выводы: рост $lambda$ $arrow.r.double$ рост разреженности (до $approx 36\%$ нулей при $lambda=1$) — soft-thresholding порождает точные нули (#lec(14)); более жёсткое $epsilon$ требует больше итераций ($O(1\/epsilon)$ для ISTA); тест-точность устойчива $approx 0.76$ (мажоритарный базис $approx 0.50$).

#figure(image("figures/prox_lambdas.png", width: 70%),
  caption: [Проксимальный градиент (ISTA), $phi.alt(W_k) - phi.alt^*$, скорость $O(1\/k)$.])
