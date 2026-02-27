import os
import textwrap
import numpy as np
from manim import *

SPEED = 1.0

def rt(t):   return t / SPEED
def wt(t):   return t / SPEED


UCI_HAR_PATH = "UCI HAR Dataset"

BG = "#0D1117"
C_BLUE = "#4A90D9"
C_GREEN = "#50C878"
C_RED = "#E74C3C"
C_GOLD = "#F0A500"
C_PURPLE = "#9B59B6"
C_TEAL = "#1ABC9C"
C_LIGHT = "#ECF0F1"
C_GRAY = "#7F8C8D"
C_ORANGE = "#E67E22"
C_PINK = "#FF6B9D"

ACT_COLORS = [C_BLUE, C_GREEN, C_RED, C_GOLD, C_PURPLE, C_TEAL]
ACT_NAMES = ["Caminar", "Subir escaleras", "Bajar escaleras",
             "Sentarse", "Pararse", "Acostarse"]


# ==========================================================
# HELPERS
# ==========================================================
def wrap(s: str, width: int = 52) -> str:
    parts = []
    for line in s.split("\n"):
        if not line.strip():
            parts.append("")
        else:
            parts.append(textwrap.fill(line, width=width))
    return "\n".join(parts)


def safe_load_ucihar(path_root: str):
    def _read_txt(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append([float(v) for v in line.strip().split()])
        return np.array(rows, dtype=np.float32)

    def _read_labels(path):
        with open(path, "r", encoding="utf-8") as f:
            return np.array([int(line.strip()) - 1 for line in f], dtype=np.int32)

    x_train_p = os.path.join(path_root, "train", "X_train.txt")
    y_train_p = os.path.join(path_root, "train", "y_train.txt")
    x_test_p = os.path.join(path_root, "test", "X_test.txt")
    y_test_p = os.path.join(path_root, "test", "y_test.txt")

    if not (os.path.exists(x_train_p) and os.path.exists(y_train_p) and
            os.path.exists(x_test_p) and os.path.exists(y_test_p)):
        raise FileNotFoundError(
            "No encuentro UCI HAR Dataset. Revisa UCI_HAR_PATH.\n"
            f"Busqué:\n- {x_train_p}\n- {y_train_p}\n- {x_test_p}\n- {y_test_p}"
        )

    X_train = _read_txt(x_train_p)
    y_train = _read_labels(y_train_p)
    X_test = _read_txt(x_test_p)
    y_test = _read_labels(y_test_p)
    return X_train, y_train, X_test, y_test


CACHE_FILE = "har_cache_models.joblib"


def train_or_load_models():
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )

    if os.path.exists(CACHE_FILE):
        data = joblib.load(CACHE_FILE)
        return data

    print("▶ Cargando UCI-HAR Dataset…")
    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = safe_load_ucihar(UCI_HAR_PATH)

    print("▶ Entrenando modelos…")
    scaler = StandardScaler()
    X_TR = scaler.fit_transform(X_TRAIN)
    X_TE = scaler.transform(X_TEST)

    clfs = {
        "Log. Reg.": LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42
        ),
    }

    results = {}
    trained = {}

    for name, clf in clfs.items():
        print(f"   Entrenando {name}…")
        clf.fit(X_TR, Y_TRAIN)
        yp = clf.predict(X_TE)
        trained[name] = clf
        results[name] = {
            "accuracy": round(accuracy_score(Y_TEST, yp) * 100, 2),
            "precision": round(precision_score(Y_TEST, yp, average="weighted", zero_division=0) * 100, 2),
            "recall": round(recall_score(Y_TEST, yp, average="weighted", zero_division=0) * 100, 2),
            "f1": round(f1_score(Y_TEST, yp, average="weighted", zero_division=0) * 100, 2),
        }
        print(f"      Accuracy: {results[name]['accuracy']}%")

    pca = PCA(n_components=2, random_state=42)
    X_PCA = pca.fit_transform(X_TE)
    pca_exp = pca.explained_variance_ratio_
    mn, mx = X_PCA.min(0), X_PCA.max(0)
    X_PCA_N = (X_PCA - mn) / (mx - mn + 1e-9) * 6 - 3

    best_model = max(results, key=lambda n: results[n]["accuracy"])
    cm_best = confusion_matrix(Y_TEST, trained[best_model].predict(X_TE), labels=list(range(6)))

    data = {
        "X_TRAIN": X_TRAIN, "Y_TRAIN": Y_TRAIN, "X_TEST": X_TEST, "Y_TEST": Y_TEST,
        "scaler": scaler, "X_TR": X_TR, "X_TE": X_TE,
        "trained": trained, "results": results,
        "X_PCA_N": X_PCA_N, "pca_exp": pca_exp,
        "best_model": best_model, "cm_best": cm_best,
    }

    joblib.dump(data, CACHE_FILE)
    return data


DATA = train_or_load_models()

X_TRAIN = DATA["X_TRAIN"]
Y_TRAIN = DATA["Y_TRAIN"]
X_TEST = DATA["X_TEST"]
Y_TEST = DATA["Y_TEST"]
RESULTS = DATA["results"]
X_PCA_N = DATA["X_PCA_N"]
PCA_EXP = DATA["pca_exp"]
BEST_MODEL = DATA["best_model"]
CM_BEST = DATA["cm_best"]

MODEL_NAMES = list(RESULTS.keys())
MODEL_COLORS = [C_BLUE, C_GREEN, C_RED, C_PURPLE, C_GOLD]


# ==========================================================
# ESCENA PRINCIPAL
# ==========================================================
class HARVideo(Scene):
    def setup(self):
        self.camera.background_color = BG
        self.FW = config.frame_width
        self.FH = config.frame_height
        self.M = 0.55
        self.HEADER_H = 0.95
        self.content_top = self.FH / 2 - self.HEADER_H - 0.15
        self.content_bottom = -self.FH / 2 + 0.45
        self.panel_h = self.content_top - self.content_bottom
        usable_w = self.FW - 3 * self.M
        self.left_w = usable_w * 0.62
        self.right_w = usable_w * 0.38
        self.panel_y = (self.content_top + self.content_bottom) / 2
        self.left_x = -self.FW / 2 + self.M + self.left_w / 2
        self.right_x = self.FW / 2 - self.M - self.right_w / 2

    # ── UI BLOCKS ──────────────────────────────────────────
    def header(self, title: str, color=C_BLUE):
        t = Text(title, font_size=28, color=color, weight=BOLD).to_edge(UP, buff=0.22)
        line = Line(LEFT * (self.FW / 2 - 0.5), RIGHT * (self.FW / 2 - 0.5),
                    color=color, stroke_width=2).next_to(t, DOWN, buff=0.10)
        return VGroup(t, line)

    def panels(self, left_title="Visualización", right_title="Paso a paso"):
        left = RoundedRectangle(
            width=self.left_w, height=self.panel_h, corner_radius=0.18,
            stroke_width=1.5, color=C_GRAY, fill_color=BG, fill_opacity=0.35
        ).move_to([self.left_x, self.panel_y, 0])
        right = RoundedRectangle(
            width=self.right_w, height=self.panel_h, corner_radius=0.18,
            stroke_width=1.5, color=C_GRAY, fill_color=BG, fill_opacity=0.35
        ).move_to([self.right_x, self.panel_y, 0])
        lt = Text(left_title, font_size=15, color=C_GRAY).next_to(left.get_top(), DOWN, buff=0.15)
        rt = Text(right_title, font_size=15, color=C_GRAY).next_to(right.get_top(), DOWN, buff=0.15)
        lt.move_to([self.left_x, lt.get_center()[1], 0])
        rt.move_to([self.right_x, rt.get_center()[1], 0])
        return VGroup(left, right, lt, rt), left, right

    def wrapped_text(self, s, font_size=16, color=C_LIGHT, width_chars=48, line_spacing=1.2):
        return Text(wrap(s, width_chars), font_size=font_size, color=color,
                    line_spacing=line_spacing)

    def step_list(self, title, steps, accent=C_BLUE):
        title_t = Text(title, font_size=17, color=accent, weight=BOLD)
        rows = VGroup()
        for i, st in enumerate(steps, 1):
            badge = Circle(radius=0.15, color=accent, stroke_width=1.5).set_fill(accent, opacity=0.18)
            num = Text(str(i), font_size=13, color=accent, weight=BOLD).move_to(badge)
            txt = self.wrapped_text(st, font_size=14, width_chars=44, color=C_LIGHT)
            row = VGroup(VGroup(badge, num), txt).arrange(RIGHT, buff=0.16, aligned_edge=UP)
            rows.add(row)
        rows.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        g = VGroup(title_t, rows).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        if g.width > self.right_w - 0.5:
            g.scale_to_fit_width(self.right_w - 0.5)
        return g

    def metric_panel(self, model_name, accent=C_BLUE):
        m = RESULTS[model_name]
        title = Text("Resultados reales (UCI-HAR)", font_size=16, color=C_LIGHT, weight=BOLD)
        bar_w = self.right_w - 1.1
        bar_h = 0.20

        def row(label, val, color):
            lab = Text(label, font_size=13, color=C_LIGHT)
            bg = Rectangle(width=bar_w, height=bar_h, stroke_width=1,
                           color=C_GRAY).set_fill(C_GRAY, 0.12)
            fill = Rectangle(width=bar_w * (val / 100.0), height=bar_h, stroke_width=0,
                             color=color).set_fill(color, 0.85).align_to(bg, LEFT)
            pct = Text(f"{val:.2f}%", font_size=13, color=color, weight=BOLD)
            top = VGroup(lab, pct).arrange(RIGHT, buff=0.22)
            bars = VGroup(bg, fill)
            return VGroup(top, bars).arrange(DOWN, buff=0.10, aligned_edge=LEFT)

        rows = VGroup(
            row("Accuracy", m["accuracy"], C_BLUE),
            row("Precision", m["precision"], C_GREEN),
            row("Recall", m["recall"], C_GOLD),
            row("F1-Score", m["f1"], C_RED),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)

        block = VGroup(title, rows).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        if block.width > self.right_w - 0.5:
            block.scale_to_fit_width(self.right_w - 0.5)
        return block

    def fade_out_all(self, t=0.6):
        mobs = list(self.mobjects)
        if mobs:
            self.play(FadeOut(Group(*mobs)), run_time=rt(t))

    def info_box(self, text, color=C_GRAY, width=None, font_size=15):
        w = width or (self.left_w - 1.4)
        box = RoundedRectangle(width=w, height=1.2, corner_radius=0.15,
                               color=color, stroke_width=1.2).set_fill(BG, 0.55)
        txt = self.wrapped_text(text, font_size=font_size, color=C_LIGHT,
                                width_chars=int(w * 6.5))
        txt.move_to(box)
        if txt.width > w - 0.3:
            txt.scale_to_fit_width(w - 0.3)
        return VGroup(box, txt)

    def construct(self):

        # ══════════════════════════════════════════════════
        # PORTADA
        # ══════════════════════════════════════════════════
        grid_lines = VGroup()
        for x in np.arange(-self.FW / 2, self.FW / 2 + 1, 1.2):
            grid_lines.add(Line([x, -self.FH / 2, 0], [x, self.FH / 2, 0],
                                color=C_GRAY, stroke_width=0.4, stroke_opacity=0.18))
        for y in np.arange(-self.FH / 2, self.FH / 2 + 1, 1.2):
            grid_lines.add(Line([-self.FW / 2, y, 0], [self.FW / 2, y, 0],
                                color=C_GRAY, stroke_width=0.4, stroke_opacity=0.18))
        self.add(grid_lines)

        top_line = Line(LEFT * (self.FW / 2 - 0.5), RIGHT * (self.FW / 2 - 0.5),
                        color=C_BLUE, stroke_width=2.5).to_edge(UP, buff=0.55)
        bot_line = Line(LEFT * (self.FW / 2 - 0.5), RIGHT * (self.FW / 2 - 0.5),
                        color=C_BLUE, stroke_width=2.5).to_edge(DOWN, buff=0.55)
        self.play(Create(top_line), Create(bot_line), run_time=rt(0.8))

        cover_title = Text(
            "Reconocimiento de\nActividades Humanas",
            font_size=44, color=C_BLUE, weight=BOLD, line_spacing=1.15
        ).move_to(ORIGIN).shift(UP * 1.6)
        self.play(Write(cover_title), run_time=rt(1.4))

        cover_sub = Text(
            "Machine Learning aplicado al dataset UCI-HAR",
            font_size=22, color=C_LIGHT
        ).next_to(cover_title, DOWN, buff=0.35)
        self.play(FadeIn(cover_sub, shift=UP * 0.1), run_time=rt(0.8))

        sep = Line(LEFT * 3.5, RIGHT * 3.5, color=C_GOLD, stroke_width=1.8) \
            .next_to(cover_sub, DOWN, buff=0.42)
        self.play(Create(sep), run_time=rt(0.6))

        integrantes_title = Text(
            "Integrantes", font_size=18, color=C_GOLD, weight=BOLD
        ).next_to(sep, DOWN, buff=0.28)
        self.play(FadeIn(integrantes_title), run_time=rt(0.5))

        nombres = [
            "Meza León, Ricardo Manuel",
            "Ramos Bonilla, Miguel Angel",
            "Cabezas Ramírez, Dylan Andres",
        ]
        nombre_items = VGroup(*[
            Text(n, font_size=19, color=C_LIGHT) for n in nombres
        ]).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        nombre_items.next_to(integrantes_title, DOWN, buff=0.22)
        nombre_items.move_to([0, nombre_items.get_center()[1], 0])

        self.play(
            LaggedStart(*[FadeIn(n, shift=RIGHT * 0.15) for n in nombre_items], lag_ratio=0.30),
            run_time=rt(1.0)
        )

        self.wait(wt(3.5))
        self.fade_out_all(0.9)

        # ══════════════════════════════════════════════════
        # SECCIÓN 1 — PORTADA + AGENDA
        # ══════════════════════════════════════════════════
        hdr = self.header("Reconocimiento de Actividades Humanas (UCI-HAR)", C_BLUE)
        self.play(Write(hdr), run_time=rt(1.4))

        subtitle = Text(
            "Machine Learning vs MLP  ·  Paso a paso con datos reales",
            font_size=21, color=C_LIGHT
        ).next_to(hdr, DOWN, buff=0.35)
        self.play(FadeIn(subtitle, shift=UP * 0.15), run_time=rt(0.9))

        context_txt = self.wrapped_text(
            "¿Qué hace tu smartphone cuando detecta que estás caminando?\n"
            "¿Cómo sabe si te caíste o si simplemente te sentaste?\n"
            "Hoy aprenderemos cómo los algoritmos de ML resuelven este problema.",
            16, C_GRAY, 66
        ).next_to(subtitle, DOWN, buff=0.45)
        self.play(FadeIn(context_txt, shift=UP * 0.1), run_time=rt(1.0))
        self.wait(wt(2.5))

        self.play(FadeOut(context_txt), run_time=rt(0.4))

        agenda_title = Text("Agenda", font_size=20, color=C_GOLD, weight=BOLD)
        agenda_items = [
            "1)  Qué es UCI-HAR: señales, ventanas y feature vectors.",
            "2)  Normalización: por qué escalar los datos importa.",
            "3)  PCA real: visualizar 561 dimensiones en 2D.",
            "4)  Regresión Logística: scores y softmax multiclase.",
            "5)  KNN: vecinos, distancias y el efecto de K.",
            "6)  SVM con kernel RBF: margen y truco del kernel.",
            "7)  Random Forest: bootstrap, splits aleatorios y voto.",
            "8)  MLP: forward pass, pérdida y backpropagación.",
            "9)  Comparación final: métricas + matriz de confusión.",
        ]
        rows = VGroup(*[self.wrapped_text(a, 15, C_LIGHT, 66) for a in agenda_items])
        rows.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        agenda_group = VGroup(agenda_title, rows).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        agenda_group.move_to(ORIGIN).shift(DOWN * 0.5)

        self.play(FadeIn(agenda_title, shift=UP * 0.1), run_time=rt(0.8))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in rows], lag_ratio=0.10),
                  run_time=rt(2.0))
        self.wait(wt(3.0))
        self.fade_out_all(0.7)

        # ══════════════════════════════════════════════════
        # SECCIÓN 2 — DATASET DETALLADO
        # ══════════════════════════════════════════════════
        hdr2 = self.header("UCI HAR Dataset — Señales, ventanas y features", C_TEAL)
        self.play(Write(hdr2), run_time=rt(1.0))
        panels_group, left_panel, right_panel = self.panels("Señal real → features", "¿Qué hay en el dataset?")
        self.play(FadeIn(panels_group), run_time=rt(0.6))

        n_train = X_TRAIN.shape[0]
        n_test = X_TEST.shape[0]
        n_feat = X_TRAIN.shape[1]

        right_info = self.step_list(
            "Estructura del dataset",
            [
                f"{n_train:,} muestras de entrenamiento, {n_test:,} de prueba.",
                f"{n_feat} features por muestra: estadísticas en tiempo y frecuencia.",
                "Cada muestra = ventana de 2.56 s (128 lecturas a 50 Hz).",
                "6 actividades: caminar, subir/bajar escaleras, sentarse, pararse, acostarse.",
                "Datos de 30 voluntarios con smartphone en la cintura.",
            ],
            accent=C_TEAL
        )
        right_info.align_to(right_panel, UL).shift(RIGHT * 0.35 + DOWN * 0.5)
        self.play(FadeIn(right_info, shift=RIGHT * 0.1), run_time=rt(0.9))

        ax_sig = Axes(
            x_range=[0, 128, 32],
            y_range=[-2.5, 2.5, 1],
            x_length=self.left_w - 1.0,
            y_length=3.0,
            axis_config={"color": C_GRAY, "stroke_width": 1.2},
        ).move_to([self.left_x, self.panel_y + 0.55, 0])

        sig_title = Text("Aceleración cruda (eje Z, una ventana de 2.56 s)", font_size=15, color=C_LIGHT) \
            .next_to(ax_sig, UP, buff=0.18)

        np.random.seed(42)
        t_vals = np.linspace(0, 128, 128)
        walking_sig = np.sin(t_vals * 2 * np.pi / 18) * 1.6 + np.random.randn(128) * 0.22

        raw_curve = ax_sig.plot_line_graph(
            list(range(128)), [float(v) for v in walking_sig],
            line_color=C_GREEN, stroke_width=2.0, add_vertex_dots=False
        )

        self.play(Create(ax_sig), FadeIn(sig_title), run_time=rt(1.0))
        self.play(Create(raw_curve), run_time=rt(1.8))

        features_box = self.info_box(
            "De esta ventana se extraen 561 features:\n"
            "media, desv. estándar, correlaciones, energía FFT, entropía…\n"
            "¡El modelo NO ve la señal cruda, ve este resumen!",
            color=C_TEAL, font_size=12
        )
        features_box.next_to(ax_sig, DOWN, buff=0.22).move_to([self.left_x, features_box.get_center()[1], 0])
        self.play(FadeIn(features_box), run_time=rt(0.8))
        self.wait(wt(2.8))

        self.play(FadeOut(VGroup(ax_sig, sig_title, raw_curve, features_box)), run_time=rt(0.5))

        unique, counts = np.unique(Y_TRAIN, return_counts=True)
        ax_dist = Axes(
            x_range=[0, 7, 1],
            y_range=[0, max(counts) + 200, 500],
            x_length=self.left_w - 1.2,
            y_length=3.5,
            axis_config={"color": C_GRAY, "stroke_width": 1.2, "include_tip": False},
            x_axis_config={"include_ticks": False},
        ).move_to([self.left_x, self.panel_y - 0.1, 0])

        dist_title = Text("Distribución de clases en entrenamiento", font_size=15, color=C_LIGHT) \
            .next_to(ax_dist, UP, buff=0.18)

        dist_bars = VGroup()
        dist_labels = VGroup()
        for i, (cls, cnt, col) in enumerate(zip(unique, counts, ACT_COLORS), start=1):
            bot = ax_dist.c2p(i - 0.33, 0)
            top = ax_dist.c2p(i + 0.33, cnt)
            rect = Rectangle(width=top[0] - bot[0], height=top[1] - bot[1],
                             color=col, stroke_width=1.2).set_fill(col, 0.75)
            rect.align_to(np.array([bot[0], bot[1], 0]), DL)
            dist_bars.add(rect)
            lbl = Text(ACT_NAMES[cls][:7], font_size=10, color=C_LIGHT)
            lbl.next_to(rect, DOWN, buff=0.08)
            dist_labels.add(lbl)

        self.play(Create(ax_dist), FadeIn(dist_title), run_time=rt(0.9))
        self.play(LaggedStart(*[DrawBorderThenFill(b) for b in dist_bars], lag_ratio=0.12), run_time=rt(1.3))
        self.play(FadeIn(dist_labels), run_time=rt(0.6))
        self.wait(wt(2.5))
        self.fade_out_all(0.7)

        # ══════════════════════════════════════════════════
        # SECCIÓN 2B — NORMALIZACIÓN
        # ══════════════════════════════════════════════════
        hdr_norm = self.header("Preprocesamiento: ¿Por qué normalizar?", C_ORANGE)
        self.play(Write(hdr_norm), run_time=rt(1.0))
        panels_group, left_panel, right_panel = self.panels("Antes vs Después", "Concepto clave")
        self.play(FadeIn(panels_group), run_time=rt(0.6))

        norm_steps = self.step_list(
            "StandardScaler — por qué usarlo",
            [
                "Cada feature tiene una escala diferente (e.g. aceleración vs ángulo).",
                "Sin escalar: KNN y SVM se ven distorsionados por features grandes.",
                "StandardScaler: substrae la media y divide por desv. estándar.",
                "Resultado: todas las features con media≈0 y σ≈1.",
            ],
            accent=C_ORANGE
        )
        norm_steps.align_to(right_panel, UL).shift(RIGHT * 0.35 + DOWN * 0.5)
        self.play(FadeIn(norm_steps), run_time=rt(0.9))

        # Visual: histograma antes/después
        f0_raw = X_TRAIN[:, 0]
        f0_sc = (f0_raw - f0_raw.mean()) / (f0_raw.std() + 1e-9)

        ax_l = Axes(x_range=[-4, 4, 2], y_range=[0, 500, 100],
                    x_length=self.left_w / 2 - 0.8, y_length=3.1,
                    axis_config={"color": C_GRAY, "stroke_width": 1.2}).move_to(
            [self.left_x - 1.4, self.panel_y + 0.1, 0])
        ax_r = Axes(x_range=[-4, 4, 2], y_range=[0, 500, 100],
                    x_length=self.left_w / 2 - 0.8, y_length=3.1,
                    axis_config={"color": C_GRAY, "stroke_width": 1.2}).move_to(
            [self.left_x + 1.4, self.panel_y + 0.1, 0])

        def make_hist(ax, data, color, bins=20):
            hist, edges = np.histogram(data, bins=bins)
            bars = VGroup()
            for cnt, e0, e1 in zip(hist, edges[:-1], edges[1:]):
                try:
                    b = ax.c2p(e0, 0)
                    t = ax.c2p(e1, float(cnt))
                    w = max(t[0] - b[0], 0.01)
                    h = max(t[1] - b[1], 0.01)
                    rect = Rectangle(width=w, height=h, color=color, stroke_width=0.5).set_fill(color, 0.7)
                    rect.align_to(np.array([b[0], b[1], 0]), DL)
                    bars.add(rect)
                except:
                    pass
            return bars

        lbl_before = Text("Antes (sin escalar)", font_size=14, color=C_RED).next_to(ax_l, UP, buff=0.15)
        lbl_after = Text("Después (normalizado)", font_size=14, color=C_GREEN).next_to(ax_r, UP, buff=0.15)

        raw_hist = make_hist(ax_l, np.clip(f0_raw, -4, 4), C_RED)
        sc_hist = make_hist(ax_r, np.clip(f0_sc, -4, 4), C_GREEN)

        self.play(Create(ax_l), Create(ax_r), FadeIn(lbl_before), FadeIn(lbl_after), run_time=rt(1.0))
        self.play(LaggedStart(*[FadeIn(b) for b in raw_hist], lag_ratio=0.02), run_time=rt(1.0))
        self.play(LaggedStart(*[FadeIn(b) for b in sc_hist], lag_ratio=0.02), run_time=rt(1.0))
        self.wait(wt(2.5))
        self.fade_out_all(0.6)

        # ══════════════════════════════════════════════════
        # SECCIÓN 3 — PCA REAL  (~65 s)
        # ══════════════════════════════════════════════════
        hdr3 = self.header("PCA: de 561 dimensiones a 2D", C_GOLD)
        self.play(Write(hdr3), run_time=rt(1.0))
        panels_group, left_panel, right_panel = self.panels("PCA Scatter (Test set)", "Cómo funciona PCA")
        self.play(FadeIn(panels_group), run_time=rt(0.6))

        v1 = round(PCA_EXP[0] * 100, 1)
        v2 = round(PCA_EXP[1] * 100, 1)

        pca_steps = self.step_list(
            "Qué hace PCA",
            [
                "Encuentra las direcciones de mayor varianza en 561 features.",
                "Proyecta cada muestra a ese plano 2D para poder visualizarla.",
                f"PC1 = {v1}% varianza  ·  PC2 = {v2}% varianza.",
                "Si clases se separan en 2D: un modelo lineal podría funcionar.",
                "Mezcla = necesitamos fronteras no lineales (SVM, RF, MLP).",
            ],
            accent=C_GOLD
        )
        pca_steps.align_to(right_panel, UL).shift(RIGHT * 0.35 + DOWN * 0.5)
        self.play(FadeIn(pca_steps), run_time=rt(0.9))

        ax = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[-3.5, 3.5, 1],
            x_length=self.left_w - 1.2,
            y_length=self.panel_h - 1.6,
            axis_config={"color": C_GRAY, "stroke_width": 1.4, "include_tip": True},
        ).move_to([self.left_x, self.panel_y - 0.1, 0])

        xlbl = Text(f"PC1 ({v1}%)", font_size=13, color=C_GRAY).next_to(ax.x_axis, RIGHT * 0.1 + DOWN * 0.25)
        ylbl = Text(f"PC2 ({v2}%)", font_size=13, color=C_GRAY).next_to(ax.y_axis, UP * 0.1 + LEFT * 0.55)
        self.play(Create(ax), FadeIn(xlbl), FadeIn(ylbl), run_time=rt(1.0))

        np.random.seed(0)
        per_class = 110
        all_groups = VGroup()
        legend_items = VGroup()

        for ci, (color, name) in enumerate(zip(ACT_COLORS, ACT_NAMES)):
            idx = np.where(Y_TEST == ci)[0]
            idx = np.random.choice(idx, min(per_class, len(idx)), replace=False)
            pts = X_PCA_N[idx]
            dots = VGroup(*[
                Dot(ax.c2p(float(x), float(y)), radius=0.035, color=color, fill_opacity=0.72)
                for x, y in pts if -3.2 < x < 3.2 and -3.2 < y < 3.2
            ])
            all_groups.add(dots)
            leg = VGroup(Dot(radius=0.055, color=color),
                         Text(name, font_size=10, color=C_LIGHT)).arrange(RIGHT, buff=0.08)
            legend_items.add(leg)

        legend = legend_items.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        legend.scale_to_fit_width(min(legend.width, self.right_w - 0.8))
        legend.next_to(pca_steps, DOWN, buff=0.28).align_to(pca_steps, LEFT)
        right_panel_bottom = right_panel.get_bottom()[1] + 0.2
        if legend.get_bottom()[1] < right_panel_bottom:
            legend.shift(UP * (right_panel_bottom - legend.get_bottom()[1]))

        for ci, grp in enumerate(all_groups):
            self.play(
                LaggedStart(*[FadeIn(d, scale=0.5) for d in grp], lag_ratio=0.005),
                FadeIn(legend_items[ci]),
                run_time=rt(0.55)
            )
        self.wait(wt(3.0))
        self.fade_out_all(0.7)

        # ══════════════════════════════════════════════════
        # SECCIÓN 4 — LOGISTIC REGRESSION
        # ══════════════════════════════════════════════════
        hdr4 = self.header("Regresión Logística — scores, softmax y frontera lineal", C_BLUE)
        self.play(Write(hdr4), run_time=rt(1.0))
        panels_group, left_panel, right_panel = self.panels("Ejemplo 2D", "Paso a paso + métricas")
        self.play(FadeIn(panels_group), run_time=rt(0.6))

        steps_lr = self.step_list(
            "Cómo decide (multiclase)",
            [
                "Paso 1: Calcula score_k = W_k · x + b_k  para cada clase k.",
                "Paso 2: Softmax convierte scores en probabilidades (suman 1.0).",
                "Paso 3: Predice la clase con mayor probabilidad.",
                "Entrena minimizando Cross-Entropy con gradiente descendente.",
                "Frontera: hiperplano (línea en 2D) entre clases.",
            ],
            accent=C_BLUE
        )
        steps_lr.align_to(right_panel, UL).shift(RIGHT * 0.35 + DOWN * 0.5)
        metrics_lr = self.metric_panel("Log. Reg.", accent=C_BLUE)
        metrics_lr.next_to(steps_lr, DOWN, buff=0.30).align_to(steps_lr, LEFT)
        self.play(FadeIn(steps_lr), run_time=rt(0.9))
        self.play(FadeIn(metrics_lr), run_time=rt(0.8))

        ax_lr = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=self.left_w - 1.4, y_length=self.panel_h - 1.9,
            axis_config={"color": C_GRAY, "stroke_width": 1.2},
        ).move_to([self.left_x, self.panel_y - 0.2, 0])

        np.random.seed(2)
        A = np.random.randn(24, 2) * 0.55 + np.array([1.1, 1.1])
        B = np.random.randn(24, 2) * 0.55 + np.array([-1.1, -1.1])
        C_pts = np.random.randn(20, 2) * 0.50 + np.array([1.1, -1.1])

        dA = VGroup(*[Dot(ax_lr.c2p(x, y), radius=0.07, color=C_GREEN, fill_opacity=0.85) for x, y in A])
        dB = VGroup(*[Dot(ax_lr.c2p(x, y), radius=0.07, color=C_RED, fill_opacity=0.85) for x, y in B])
        dC = VGroup(*[Dot(ax_lr.c2p(x, y), radius=0.07, color=C_GOLD, fill_opacity=0.85) for x, y in C_pts])

        boundary1 = ax_lr.plot(lambda x: -x + 0.2, color=C_BLUE, stroke_width=2.5)
        boundary2 = ax_lr.plot(lambda x: x + 0.1, color=C_PURPLE, stroke_width=2.5)

        q = Dot(ax_lr.c2p(0.3, 0.1), radius=0.11, color=C_LIGHT)
        q_lbl = Text("x nueva", font_size=13, color=C_LIGHT).next_to(q, UR, buff=0.08)

        self.play(Create(ax_lr), run_time=rt(0.8))
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.6) for d in dA], lag_ratio=0.04),
            LaggedStart(*[FadeIn(d, scale=0.6) for d in dB], lag_ratio=0.04),
            LaggedStart(*[FadeIn(d, scale=0.6) for d in dC], lag_ratio=0.04),
            run_time=rt(1.3)
        )
        self.play(Create(boundary1), Create(boundary2), run_time=rt(1.0))
        self.play(FadeIn(q), FadeIn(q_lbl), run_time=rt(0.6))

        # Softmax mini-demo
        scores_txt = self.info_box(
            "score_A=2.1  score_B=-0.5  score_C=0.8\n"
            "Softmax → p(A)=0.76  p(B)=0.05  p(C)=0.19\n"
            "→  Predicción: Clase A  (mayor probabilidad)",
            color=C_BLUE, font_size=14
        )
        scores_txt.next_to(ax_lr, DOWN, buff=0.22).move_to([self.left_x, scores_txt.get_center()[1], 0])
        self.play(FadeIn(scores_txt), run_time=rt(0.7))
        self.wait(wt(3.0))
        self.fade_out_all(0.7)

        # ══════════════════════════════════════════════════
        # SECCIÓN 5 — KNN
        # ══════════════════════════════════════════════════
        hdr5 = self.header("KNN — vecinos, distancias y el efecto de K", C_GREEN)
        self.play(Write(hdr5), run_time=rt(1.0))
        panels_group, left_panel, right_panel = self.panels("Vecinos en 2D", "Paso a paso + métricas")
        self.play(FadeIn(panels_group), run_time=rt(0.6))

        steps_knn = self.step_list(
            "Cómo decide",
            [
                "Calcula distancia euclídea (o coseno) a TODOS los puntos de train.",
                "Selecciona los K puntos más cercanos a la consulta.",
                "Cuenta votos por clase entre esos K vecinos.",
                "K pequeño = más ruido; K grande = más suave pero menos preciso.",
                "En UCI-HAR usamos K=7 (menos overfitting que K=1 o K=3).",
            ],
            accent=C_GREEN
        )
        steps_knn.align_to(right_panel, UL).shift(RIGHT * 0.35 + DOWN * 0.5)
        max_step_h = (self.panel_h * 0.52)
        if steps_knn.height > max_step_h:
            steps_knn.scale(max_step_h / steps_knn.height)
            steps_knn.align_to(right_panel, UL).shift(RIGHT * 0.35 + DOWN * 0.5)

        metrics_knn = self.metric_panel("KNN", accent=C_GREEN)
        metrics_knn.next_to(steps_knn, DOWN, buff=0.22).align_to(steps_knn, LEFT)
        rp_bottom = right_panel.get_bottom()[1] + 0.2
        if metrics_knn.height > (self.panel_h * 0.42):
            metrics_knn.scale((self.panel_h * 0.42) / metrics_knn.height)
        if metrics_knn.get_bottom()[1] < rp_bottom:
            shift_up = rp_bottom - metrics_knn.get_bottom()[1]
            metrics_knn.shift(UP * shift_up)
            steps_knn.shift(UP * shift_up)

        self.play(FadeIn(steps_knn), FadeIn(metrics_knn), run_time=rt(0.9))

        ax_knn = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=self.left_w - 1.4, y_length=self.panel_h - 2.0,
            axis_config={"color": C_GRAY, "stroke_width": 1.2},
        ).move_to([self.left_x, self.panel_y, 0])

        np.random.seed(10)
        pts_a = np.random.randn(28, 2) * 0.70 + np.array([0.9, 0.7])
        pts_b = np.random.randn(28, 2) * 0.70 + np.array([-0.9, -0.7])

        dots_a = VGroup(*[Dot(ax_knn.c2p(x, y), radius=0.065, color=C_BLUE, fill_opacity=0.85) for x, y in pts_a])
        dots_b = VGroup(*[Dot(ax_knn.c2p(x, y), radius=0.065, color=C_RED, fill_opacity=0.85) for x, y in pts_b])

        qxy = np.array([0.15, 0.15])
        qdot = Dot(ax_knn.c2p(qxy[0], qxy[1]), radius=0.11, color=C_GOLD)
        qlab = Text("consulta", font_size=13, color=C_GOLD).next_to(qdot, UR, buff=0.09)

        self.play(Create(ax_knn), run_time=rt(0.8))
        self.play(FadeIn(dots_a), FadeIn(dots_b), run_time=rt(1.0))
        self.play(FadeIn(qdot), FadeIn(qlab), run_time=rt(0.5))

        all_pts = np.vstack([pts_a, pts_b])
        all_cls = np.array([0] * len(pts_a) + [1] * len(pts_b))

        def show_knn_k(k, color=C_GOLD):
            dists = np.linalg.norm(all_pts - qxy[None, :], axis=1)
            nn = np.argsort(dists)[:k]
            lines = VGroup(*[
                Line(ax_knn.c2p(qxy[0], qxy[1]),
                     ax_knn.c2p(all_pts[i, 0], all_pts[i, 1]),
                     color=color, stroke_width=2.0, stroke_opacity=0.55)
                for i in nn
            ])
            rings = VGroup(*[
                Circle(radius=0.13, color=color, stroke_width=2.2).move_to(
                    ax_knn.c2p(all_pts[i, 0], all_pts[i, 1]))
                for i in nn
            ])
            votes_a = int(np.sum(all_cls[nn] == 0))
            votes_b = int(np.sum(all_cls[nn] == 1))
            pred = "Azul" if votes_a >= votes_b else "Rojo"
            pred_col = C_BLUE if votes_a >= votes_b else C_RED
            vote_txt = self.wrapped_text(
                f"K = {k}   |   Azul: {votes_a}   Rojo: {votes_b}\n→ Predicción: {pred}",
                font_size=15, color=pred_col, width_chars=32
            )
            box = RoundedRectangle(width=3.1, height=1.1, corner_radius=0.14,
                                   color=C_GRAY, stroke_width=1.2).set_fill(BG, 0.60)
            box.next_to(ax_knn, UP, buff=0.15).align_to(ax_knn, RIGHT).shift(LEFT * 0.1)
            top_limit = self.FH / 2 - 1.4
            if box.get_top()[1] > top_limit:
                box.shift(DOWN * (box.get_top()[1] - top_limit))
            vote_txt.move_to(box)
            return VGroup(lines, rings, box, vote_txt)

        for k in [1, 3, 7, 15]:
            vis = show_knn_k(k)
            self.play(Create(vis[0]), FadeIn(vis[1]), FadeIn(vis[2]), FadeIn(vis[3]), run_time=rt(0.9))
            self.wait(wt(1.8))
            self.play(FadeOut(vis), run_time=rt(0.4))

        self.wait(wt(1.0))
        self.fade_out_all(0.7)

        # ══════════════════════════════════════════════════
        # SECCIÓN 6 — SVM
        # ══════════════════════════════════════════════════
        hdr6 = self.header("SVM con kernel RBF — margen máximo y truco del kernel", C_RED)
        self.play(Write(hdr6), run_time=rt(1.0))
        panels_group, left_panel, right_panel = self.panels("Margen y soporte", "Paso a paso + métricas")
        self.play(FadeIn(panels_group), run_time=rt(0.6))

        steps_svm = self.step_list(
            "Cómo decide (intuición)",
            [
                "Busca la frontera con el MÁXIMO margen entre las dos clases.",
                "Solo los puntos más cercanos definen la solución: vectores soporte.",
                "Con kernel RBF: mide similitud por distancia (puntos cercanos = más influencia).",
                "C grande = ajuste fuerte (riesgo overfitting). γ pequeño = frontera más suave.",
                "En UCI-HAR: C=10, gamma='scale' → frontera no lineal de alto rendimiento.",
            ],
            accent=C_RED
        )
        steps_svm.align_to(right_panel, UL).shift(RIGHT * 0.35 + DOWN * 0.5)
        max_step_h_svm = (self.panel_h * 0.52)
        if steps_svm.height > max_step_h_svm:
            steps_svm.scale(max_step_h_svm / steps_svm.height)
            steps_svm.align_to(right_panel, UL).shift(RIGHT * 0.35 + DOWN * 0.5)

        metrics_svm = self.metric_panel("SVM", accent=C_RED)
        metrics_svm.next_to(steps_svm, DOWN, buff=0.22).align_to(steps_svm, LEFT)
        rp_bottom_svm = right_panel.get_bottom()[1] + 0.2
        if metrics_svm.height > (self.panel_h * 0.42):
            metrics_svm.scale((self.panel_h * 0.42) / metrics_svm.height)
        if metrics_svm.get_bottom()[1] < rp_bottom_svm:
            shift_up = rp_bottom_svm - metrics_svm.get_bottom()[1]
            metrics_svm.shift(UP * shift_up)
            steps_svm.shift(UP * shift_up)

        self.play(FadeIn(steps_svm), FadeIn(metrics_svm), run_time=rt(0.9))

        ax_svm = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=self.left_w - 1.4, y_length=self.panel_h - 2.0,
            axis_config={"color": C_GRAY, "stroke_width": 1.2},
        ).move_to([self.left_x, self.panel_y, 0])

        np.random.seed(5)
        pts_sa = np.random.randn(18, 2) * 0.55 + np.array([1.3, 1.1])
        pts_sb = np.random.randn(18, 2) * 0.55 + np.array([-1.3, -1.1])

        ds_a = VGroup(*[Dot(ax_svm.c2p(x, y), radius=0.065, color=C_BLUE, fill_opacity=0.85) for x, y in pts_sa])
        ds_b = VGroup(*[Dot(ax_svm.c2p(x, y), radius=0.065, color=C_RED, fill_opacity=0.85) for x, y in pts_sb])

        boundary_s = ax_svm.plot(lambda x: -x, color=C_RED, stroke_width=3.0)
        margin_up = ax_svm.plot(lambda x: -x + 1.1, color=C_RED, stroke_width=1.5, stroke_opacity=0.35)
        margin_down = ax_svm.plot(lambda x: -x - 1.1, color=C_RED, stroke_width=1.5, stroke_opacity=0.35)

        # Relleno de margen
        margin_area = ax_svm.get_area(
            ax_svm.plot(lambda x: -x + 1.1),
            bounded_graph=ax_svm.plot(lambda x: -x - 1.1),
            color=C_RED, opacity=0.07
        )

        sv_idx_a = np.argsort(np.abs(pts_sa[:, 0] + pts_sa[:, 1]))[:3]
        sv_idx_b = np.argsort(np.abs(pts_sb[:, 0] + pts_sb[:, 1]))[:3]
        sv_circles = VGroup(
            *[Circle(radius=0.13, color=C_GOLD, stroke_width=2.2).move_to(ax_svm.c2p(pts_sa[i, 0], pts_sa[i, 1])) for i
              in sv_idx_a],
            *[Circle(radius=0.13, color=C_GOLD, stroke_width=2.2).move_to(ax_svm.c2p(pts_sb[i, 0], pts_sb[i, 1])) for i
              in sv_idx_b],
        )
        sv_lbl = Text("Vectores soporte (definen el margen)", font_size=13, color=C_GOLD) \
            .next_to(ax_svm, UP, buff=0.15).align_to(ax_svm, LEFT)

        self.play(Create(ax_svm), run_time=rt(0.8))
        self.play(FadeIn(ds_a), FadeIn(ds_b), run_time=rt(1.0))
        self.play(Create(boundary_s), run_time=rt(0.8))
        self.play(Create(margin_up), Create(margin_down), FadeIn(margin_area), run_time=rt(0.8))
        self.play(FadeIn(sv_circles), FadeIn(sv_lbl), run_time=rt(0.7))

        # Kernel RBF explicación visual
        kernel_box = self.info_box(
            "Kernel RBF:\n"
            "K(x_i, x_j) = exp(-gamma ||x_i - x_j||^2)\n"
            "Mide similitud: puntos cercanos -> valor alto\n"
            "lejanos -> ~0",
            color=C_RED,
            font_size=13
        )
        kernel_box.next_to(ax_svm, DOWN, buff=0.18).move_to([self.left_x, kernel_box.get_center()[1], 0])
        lp_bottom = left_panel.get_bottom()[1] + 0.15
        if kernel_box.get_bottom()[1] < lp_bottom:
            kernel_box.shift(UP * (lp_bottom - kernel_box.get_bottom()[1]))
        self.play(FadeIn(kernel_box), run_time=rt(0.8))
        self.wait(wt(3.2))
        self.fade_out_all(0.7)

        # ══════════════════════════════════════════════════
        # SECCIÓN 7 — RANDOM FOREST
        # ══════════════════════════════════════════════════

        hdr7 = self.header(
            "Random Forest — bootstrap, splits y voto de árboles",
            C_PURPLE
        )
        self.play(Write(hdr7), run_time=rt(1.0))

        panels_group, left_panel, right_panel = self.panels(
            "Ensemble visual",
            "Paso a paso + métricas"
        )
        self.play(FadeIn(panels_group), run_time=rt(0.6))

        # PANEL DERECHO
        steps_rf = self.step_list(
            "Cómo funciona",
            [
                "Bootstrap: cada árbol entrena con muestra CON REEMPLAZO del dataset.",
                "En cada nodo se eligen √(n_features) features al azar.",
                "Cada árbol crece independientemente.",
                "El bosque vota: gana la clase mayoritaria.",
                "Feature importance reduce impureza.",
            ],
            accent=C_PURPLE
        )
        steps_rf.align_to(right_panel, UL).shift(RIGHT * 0.35 + DOWN * 0.5)
        max_step_h_rf = (self.panel_h * 0.52)
        if steps_rf.height > max_step_h_rf:
            steps_rf.scale(max_step_h_rf / steps_rf.height)
            steps_rf.align_to(right_panel, UL).shift(RIGHT * 0.35 + DOWN * 0.5)

        metrics_rf = self.metric_panel("Random Forest", accent=C_PURPLE)
        metrics_rf.next_to(steps_rf, DOWN, buff=0.22).align_to(steps_rf, LEFT)
        rp_bottom_rf = right_panel.get_bottom()[1] + 0.2
        if metrics_rf.height > (self.panel_h * 0.42):
            metrics_rf.scale((self.panel_h * 0.42) / metrics_rf.height)
        if metrics_rf.get_bottom()[1] < rp_bottom_rf:
            shift_up_rf = rp_bottom_rf - metrics_rf.get_bottom()[1]
            metrics_rf.shift(UP * shift_up_rf)
            steps_rf.shift(UP * shift_up_rf)

        self.play(FadeIn(steps_rf), FadeIn(metrics_rf), run_time=rt(1.0))

        lp_cx = left_panel.get_center()[0]
        lp_top_y = left_panel.get_top()[1] - 0.5
        lp_bot_y = left_panel.get_bottom()[1] + 0.3

        total_h = lp_top_y - lp_bot_y
        row1_y = lp_top_y - 0.6
        row2_y = row1_y - total_h * 0.30
        row3_y = row2_y - total_h * 0.28
        row4_y = lp_bot_y + 0.5

        data_box = RoundedRectangle(
            width=2.0, height=1.0, corner_radius=0.15, color=C_GRAY
        ).move_to([lp_cx, row1_y, 0])
        data_lbl = Text("Dataset original", font_size=13, color=C_LIGHT).move_to(data_box)
        self.play(FadeIn(data_box), FadeIn(data_lbl), run_time=rt(0.8))

        tree_colors = [C_BLUE, C_GREEN, C_GOLD]
        n_trees = 3
        tree_xs = np.linspace(lp_cx - 2.0, lp_cx + 2.0, n_trees)

        boot_groups = VGroup()
        boot_arrows = VGroup()

        for t in range(n_trees):
            cx = tree_xs[t]
            box = RoundedRectangle(
                width=1.5, height=0.85, corner_radius=0.12, color=tree_colors[t]
            ).move_to([cx, row2_y, 0])
            lbl = Text(f"Bootstrap {t + 1}", font_size=11, color=C_LIGHT).move_to(box)
            arrow = Arrow(
                data_box.get_bottom(),
                box.get_top(),
                color=tree_colors[t], stroke_width=2.5, buff=0.05
            )
            boot_groups.add(VGroup(box, lbl))
            boot_arrows.add(arrow)

        for i in range(n_trees):
            self.play(GrowArrow(boot_arrows[i]), FadeIn(boot_groups[i]), run_time=rt(0.55))

        tree_groups = VGroup()
        tree_arrows2 = VGroup()

        for t in range(n_trees):
            cx = tree_xs[t]
            box = RoundedRectangle(
                width=1.4, height=0.75, corner_radius=0.12, color=tree_colors[t]
            ).move_to([cx, row3_y, 0])
            lbl = Text(f"Árbol {t + 1}", font_size=13, color=C_LIGHT).move_to(box)
            arrow = Arrow(
                boot_groups[t].get_bottom(),
                box.get_top(),
                color=tree_colors[t], stroke_width=2.5, buff=0.05
            )
            tree_groups.add(VGroup(box, lbl))
            tree_arrows2.add(arrow)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in tree_arrows2]),
            FadeIn(tree_groups),
            run_time=rt(0.8)
        )

        vote_box = RoundedRectangle(
            width=4.5, height=0.75, corner_radius=0.15, color=C_GOLD
        ).move_to([lp_cx, row4_y, 0])
        vote_lbl = Text("Voto mayoritario → Clase final", font_size=14, color=C_GOLD).move_to(vote_box)

        final_arrows = VGroup(*[
            Arrow(
                tree_groups[i].get_bottom(),
                vote_box.get_top(),
                color=C_GOLD, stroke_width=2.5, buff=0.05
            )
            for i in range(n_trees)
        ])

        self.play(LaggedStart(*[GrowArrow(a) for a in final_arrows]), run_time=rt(0.7))
        self.play(FadeIn(vote_box), FadeIn(vote_lbl), run_time=rt(0.6))

        self.wait(wt(3.0))
        self.fade_out_all()

        # ══════════════════════════════════════════════════
        # SECCIÓN 8 — MLP
        # ══════════════════════════════════════════════════
        hdr8 = self.header("MLP (Red Neuronal) — forward, pérdida y backpropagación", C_GOLD)
        self.play(Write(hdr8), run_time=rt(1.0))
        panels_group, left_panel, right_panel = self.panels("Red neuronal animada", "Paso a paso + métricas")
        self.play(FadeIn(panels_group), run_time=rt(0.6))

        steps_mlp = self.step_list(
            "Cómo aprende y decide",
            [
                "Forward: z = W·x + b, luego aplica ReLU(z) = max(0,z) capa por capa.",
                "Capa de salida: Softmax produce probabilidades para las 6 clases.",
                "Loss: Cross-Entropy = −log(p_clase_real). Mide cuán equivocado estaba.",
                "Backprop: derivada de la pérdida w.r.t. cada peso usando regla de la cadena.",
                "Adam optimizer: actualiza pesos adaptativamente. 200 épocas, early stopping.",
            ],
            accent=C_GOLD
        )
        steps_mlp.align_to(right_panel, UL).shift(RIGHT * 0.35 + DOWN * 0.5)
        max_step_h_mlp = (self.panel_h * 0.50)
        if steps_mlp.height > max_step_h_mlp:
            steps_mlp.scale(max_step_h_mlp / steps_mlp.height)
            steps_mlp.align_to(right_panel, UL).shift(RIGHT * 0.35 + DOWN * 0.5)

        metrics_mlp = self.metric_panel("MLP", accent=C_GOLD)
        metrics_mlp.next_to(steps_mlp, DOWN, buff=0.22).align_to(steps_mlp, LEFT)
        rp_bottom_mlp = right_panel.get_bottom()[1] + 0.2
        if metrics_mlp.height > (self.panel_h * 0.42):
            metrics_mlp.scale((self.panel_h * 0.42) / metrics_mlp.height)
        if metrics_mlp.get_bottom()[1] < rp_bottom_mlp:
            shift_up_mlp = rp_bottom_mlp - metrics_mlp.get_bottom()[1]
            metrics_mlp.shift(UP * shift_up_mlp)
            steps_mlp.shift(UP * shift_up_mlp)

        self.play(FadeIn(steps_mlp), FadeIn(metrics_mlp), run_time=rt(0.9))

        # ── Arquitectura MLP ────────────────────────────
        arch = [4, 6, 5, 4, 3]  # mini red ilustrativa
        x_pos = np.linspace(-2.5, 2.5, len(arch))
        node_colors_mlp = [C_TEAL, C_BLUE, C_BLUE, C_BLUE, C_GOLD]
        layer_labels = ["Input\n(features)", "Hidden 1\nReLU", "Hidden 2\nReLU", "Hidden 3\nReLU", "Output\nSoftmax"]

        net_center_y = self.panel_y + 0.2

        nodes = []
        for li, (n, col) in enumerate(zip(arch, node_colors_mlp)):
            ys = np.linspace(-(n - 1) * 0.38, (n - 1) * 0.38, n)
            layer = VGroup(*[
                Circle(radius=0.14, color=col, stroke_width=1.8).set_fill(col, 0.22)
                           .move_to([self.left_x + x_pos[li], net_center_y + y, 0])
                for y in ys
            ])
            nodes.append(layer)

        edges = VGroup()
        for li in range(len(nodes) - 1):
            for a in nodes[li]:
                for b in nodes[li + 1]:
                    edges.add(Line(a.get_center(), b.get_center(),
                                   color=C_GRAY, stroke_width=0.55, stroke_opacity=0.22))

        arch_labels_mlp = VGroup()
        for li, (lbl_txt, col) in enumerate(zip(layer_labels, node_colors_mlp)):
            t = Text(lbl_txt, font_size=10, color=col, line_spacing=1.0)
            t.next_to(nodes[li], DOWN, buff=0.18)
            arch_labels_mlp.add(t)

        net_title = Text("Arquitectura MLP (256→128→64 en el modelo real)", font_size=13, color=C_LIGHT)
        net_title.next_to(VGroup(*nodes), UP, buff=0.20)
        lp_top_limit = left_panel.get_top()[1] - 0.2
        if net_title.get_top()[1] > lp_top_limit:
            net_title.shift(DOWN * (net_title.get_top()[1] - lp_top_limit))

        self.play(FadeIn(net_title), Create(edges), run_time=rt(1.0))
        self.play(LaggedStart(*[FadeIn(layer) for layer in nodes], lag_ratio=0.15), run_time=rt(1.0))
        self.play(FadeIn(arch_labels_mlp), run_time=rt(0.6))

        pulse_lbl = Text("Forward pass →", font_size=13, color=C_GRAY)
        pulse_lbl.next_to(arch_labels_mlp, DOWN, buff=0.25)
        lp_bot_limit = left_panel.get_bottom()[1] + 0.3
        if pulse_lbl.get_bottom()[1] < lp_bot_limit:
            pulse_lbl.shift(UP * (lp_bot_limit - pulse_lbl.get_bottom()[1]))

        self.play(FadeIn(pulse_lbl), run_time=rt(0.4))
        for _ in range(2):
            for layer in nodes:
                self.play(*[n.animate.set_fill(C_GOLD, opacity=0.90) for n in layer], run_time=rt(0.15))
                self.play(*[n.animate.set_fill(n.get_color(), opacity=0.22) for n in layer], run_time=rt(0.14))

        # Backprop animado (pulso inverso en rojo)
        bp_lbl = Text("← Backpropagación", font_size=13, color=C_RED)
        bp_lbl.next_to(pulse_lbl, DOWN, buff=0.14)
        if bp_lbl.get_bottom()[1] < lp_bot_limit:
            bp_lbl.shift(UP * (lp_bot_limit - bp_lbl.get_bottom()[1]))

        self.play(FadeIn(bp_lbl), run_time=rt(0.4))
        for _ in range(2):
            for layer in reversed(nodes):
                self.play(*[n.animate.set_fill(C_RED, opacity=0.80) for n in layer], run_time=rt(0.15))
                self.play(*[n.animate.set_fill(n.get_color(), opacity=0.22) for n in layer], run_time=rt(0.14))

        self.wait(wt(1.5))

        # Curva de pérdida animada
        self.play(FadeOut(VGroup(edges, *nodes, arch_labels_mlp, net_title, pulse_lbl, bp_lbl)),
                  run_time=rt(0.5))

        ax_loss = Axes(
            x_range=[0, 50, 10],
            y_range=[0.0, 1.1, 0.2],
            x_length=self.left_w - 1.6,
            y_length=3.5,
            axis_config={"color": C_GRAY, "stroke_width": 1.2},
        ).move_to([self.left_x, self.panel_y + 0.1, 0])

        loss_title = Text("Curva de pérdida (simulada) durante el entrenamiento", font_size=14, color=C_LIGHT) \
            .next_to(ax_loss, UP, buff=0.18)
        xlbl_l = Text("Épocas", font_size=13, color=C_GRAY).next_to(ax_loss.x_axis, RIGHT * 0.1 + DOWN * 0.22)
        ylbl_l = Text("Loss", font_size=13, color=C_GRAY).next_to(ax_loss.y_axis, UP * 0.1 + LEFT * 0.4)

        epochs = list(range(0, 51, 5))
        train_loss = [1.0, 0.82, 0.68, 0.56, 0.47, 0.40, 0.35, 0.30, 0.26, 0.23, 0.21]
        val_loss = [1.0, 0.85, 0.72, 0.61, 0.53, 0.47, 0.42, 0.38, 0.35, 0.33, 0.32]

        train_graph = ax_loss.plot_line_graph(epochs, train_loss, line_color=C_GOLD, stroke_width=3,
                                              add_vertex_dots=True)
        val_graph = ax_loss.plot_line_graph(epochs, val_loss, line_color=C_PURPLE, stroke_width=2, add_vertex_dots=True)

        legend_loss = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.4, color=C_GOLD, stroke_width=3),
                   Text("Train loss", font_size=13, color=C_GOLD)).arrange(RIGHT, buff=0.12),
            VGroup(Line(ORIGIN, RIGHT * 0.4, color=C_PURPLE, stroke_width=2),
                   Text("Val loss", font_size=13, color=C_PURPLE)).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        legend_loss.next_to(ax_loss, UP, buff=0.12).align_to(ax_loss, RIGHT)
        mlp_left = left_panel.get_left()[0] + 0.2
        mlp_right = left_panel.get_right()[0] - 0.2
        mlp_top = left_panel.get_top()[1] - 0.2
        if legend_loss.get_left()[0] < mlp_left:
            legend_loss.shift(RIGHT * (mlp_left - legend_loss.get_left()[0]))
        if legend_loss.get_right()[0] > mlp_right:
            legend_loss.shift(LEFT * (legend_loss.get_right()[0] - mlp_right))
        if legend_loss.get_top()[1] > mlp_top:
            legend_loss.shift(DOWN * (legend_loss.get_top()[1] - mlp_top))

        self.play(Create(ax_loss), FadeIn(loss_title), FadeIn(xlbl_l), FadeIn(ylbl_l), run_time=rt(0.9))
        self.play(Create(train_graph), run_time=rt(1.5))
        self.play(Create(val_graph), run_time=rt(1.5))
        self.play(FadeIn(legend_loss), run_time=rt(0.6))

        relu_box = self.info_box(
            "ReLU(z) = max(0, z)  →  añade no-linealidad capa a capa.\n"
            "Sin activación, una red profunda sería solo una transformación lineal.",
            color=C_GOLD, font_size=14
        )
        relu_box.next_to(ax_loss, DOWN, buff=0.22).move_to([self.left_x, relu_box.get_center()[1], 0])
        self.play(FadeIn(relu_box), run_time=rt(0.7))
        self.wait(wt(3.0))
        self.fade_out_all(0.7)

        # ══════════════════════════════════════════════════
        # SECCIÓN 9 — COMPARACIÓN FINAL
        # ══════════════════════════════════════════════════
        hdr9 = self.header("Comparación final — métricas reales (UCI-HAR)", C_TEAL)
        self.play(Write(hdr9), run_time=rt(1.0))
        panels_group, left_panel, right_panel = self.panels("Accuracy por modelo", "Matriz de confusión (mejor)")
        self.play(FadeIn(panels_group), run_time=rt(0.6))

        accs = [RESULTS[n]["accuracy"] for n in MODEL_NAMES]
        min_a = max(0, min(accs) - 5)

        ax_bar = Axes(
            x_range=[0, len(MODEL_NAMES) + 1, 1],
            y_range=[min_a, 100, 5],
            x_length=self.left_w - 1.2,
            y_length=self.panel_h - 1.7,
            axis_config={"color": C_GRAY, "stroke_width": 1.2, "include_tip": False},
            x_axis_config={"include_ticks": False},
        ).move_to([self.left_x, self.panel_y - 0.05, 0])

        self.play(Create(ax_bar), run_time=rt(0.9))

        bars = VGroup()
        labels = VGroup()
        short_names = ["LR", "KNN", "SVM", "RF", "MLP"]
        for i, (name, sh, acc, col) in enumerate(zip(MODEL_NAMES, short_names, accs, MODEL_COLORS), start=1):
            bot = ax_bar.c2p(i - 0.34, min_a)
            top = ax_bar.c2p(i + 0.34, acc)
            rect = Rectangle(width=top[0] - bot[0], height=top[1] - bot[1],
                             color=col, stroke_width=1.2).set_fill(col, 0.78)
            rect.align_to(np.array([bot[0], bot[1], 0]), DL)
            bars.add(rect)
            t = Text(f"{sh}\n{acc:.2f}%", font_size=12, color=C_LIGHT, line_spacing=1.1)
            t.next_to(rect, DOWN, buff=0.10)
            labels.add(t)

        self.play(LaggedStart(*[DrawBorderThenFill(b) for b in bars], lag_ratio=0.15), run_time=rt(1.5))
        self.play(FadeIn(labels), run_time=rt(0.7))

        best_note = Text(
            f"Mejor: {BEST_MODEL} — {RESULTS[BEST_MODEL]['accuracy']:.2f}% Accuracy",
            font_size=16, color=C_GOLD, weight=BOLD
        ).next_to(ax_bar, UP, buff=0.22)
        self.play(FadeIn(best_note), run_time=rt(0.6))
        self.wait(wt(2.5))

        # Confusion matrix (best)
        cm = CM_BEST.astype(int)
        maxv = max(1, cm.max())
        cell_size = min((self.right_w - 0.8) / 6.0, 0.46)

        cells = VGroup()
        for r in range(6):
            for c in range(6):
                val = cm[r, c]
                opacity = 0.08 + 0.80 * (val / maxv)
                cell = Square(side_length=cell_size, stroke_width=1.0, color=C_GRAY)
                cell.set_fill(C_TEAL if r == c else C_GRAY, opacity=opacity)
                txt = Text(str(val), font_size=12, color=C_LIGHT)
                txt.move_to(cell)
                grp = VGroup(cell, txt)
                grp.move_to([
                    self.right_x - (self.right_w / 2) + cell_size * (c + 0.5) + 0.3,
                    self.panel_y + (3 - r - 0.5) * cell_size,
                    0
                ])
                cells.add(grp)

        abbrev = ["C", "↑", "↓", "S", "P", "A"]
        row_lbls = VGroup(*[
            Text(abbrev[r], font_size=11, color=ACT_COLORS[r])
                          .next_to(cells[r * 6], LEFT, buff=0.1)
            for r in range(6)
        ])
        col_lbls = VGroup(*[
            Text(abbrev[c], font_size=11, color=ACT_COLORS[c])
                          .next_to(cells[c], UP, buff=0.1)
            for c in range(6)
        ])

        cm_title = Text("Matriz de confusión", font_size=16, color=C_LIGHT, weight=BOLD)
        cm_sub = self.wrapped_text(
            "Diagonal = predicciones correctas.\nFuera diagonal = confusiones entre actividades.",
            font_size=13, color=C_GRAY, width_chars=42
        )
        cm_header = VGroup(cm_title, cm_sub).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        cm_header.align_to(right_panel, UL).shift(RIGHT * 0.35 + DOWN * 0.5)

        self.play(FadeIn(cm_header), run_time=rt(0.7))
        cells_and_labels = VGroup(cells, row_lbls, col_lbls)
        cells_and_labels.next_to(cm_header, DOWN, buff=0.28).align_to(cm_header, LEFT)

        self.play(LaggedStart(*[FadeIn(c, scale=0.8) for c in cells], lag_ratio=0.01), run_time=rt(1.5))
        self.play(FadeIn(row_lbls), FadeIn(col_lbls), run_time=rt(0.6))
        self.wait(wt(3.0))
        self.fade_out_all(0.7)

        # ══════════════════════════════════════════════════
        # SECCIÓN 10 — CONCLUSIONES
        # ══════════════════════════════════════════════════
        hdr10 = self.header("Conclusiones y reflexiones finales", C_GOLD)
        self.play(Write(hdr10), run_time=rt(1.0))

        conclusiones = [
            (f"1) Con features bien diseñadas, TODOS los modelos rinden alto en UCI-HAR.\n"
             f"   Mejor resultado: {BEST_MODEL} con {RESULTS[BEST_MODEL]['accuracy']:.2f}% Accuracy.",
             C_GOLD),
            ("2) Logistic Regression: simple, interpretable, funciona si hay separabilidad lineal.",
             C_BLUE),
            ("3) KNN: intuitivo, pero lento en inferencia (busca en todo el dataset). K importa.",
             C_GREEN),
            ("4) SVM+RBF: potente con datos medianos. El kernel hace el trabajo duro.",
             C_RED),
            ("5) Random Forest: robusto al ruido, da importancia de features, fácil de paralelizar.",
             C_PURPLE),
            ("6) MLP: aprende sus propias representaciones; necesita más datos y más ajuste.",
             C_GOLD),
            ("7) En problemas reales: probar varios modelos, comparar métricas, NO solo accuracy.",
             C_TEAL),
        ]

        con_items = VGroup(*[
            self.wrapped_text(txt, 15, col, 72) for txt, col in conclusiones
        ])
        con_items.arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        con_items.scale_to_fit_height(self.FH - 2.0)
        con_items.move_to(ORIGIN).shift(DOWN * 0.3)

        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.1) for c in con_items], lag_ratio=0.14),
                  run_time=rt(2.2))
        self.wait(wt(4.0))

        # Fade a pantalla de cierre
        self.play(FadeOut(con_items, shift=UP * 0.1), run_time=rt(0.8))

        # Pantalla final
        final_title = Text("¡Gracias por ver!", font_size=36, color=C_BLUE, weight=BOLD)
        final_sub = Text("UCI-HAR · ML vs MLP · Animaciones paso a paso", font_size=20, color=C_LIGHT)
        final_acc_line = Text(
            f"Mejor modelo: {BEST_MODEL}  |  Accuracy: {RESULTS[BEST_MODEL]['accuracy']:.2f}%",
            font_size=18, color=C_GOLD
        )
        final_group = VGroup(final_title, final_sub, final_acc_line) \
            .arrange(DOWN, buff=0.40).move_to(ORIGIN)

        self.play(Write(final_title), run_time=rt(1.2))
        self.play(FadeIn(final_sub, shift=UP * 0.15), run_time=rt(0.8))
        self.play(FadeIn(final_acc_line, shift=UP * 0.1), run_time=rt(0.8))
        self.wait(wt(3.5))
        self.fade_out_all(0.8)
