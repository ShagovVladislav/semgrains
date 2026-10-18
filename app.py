from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from semgrains.config import Config
from semgrains.io import load_image, detect_crop, propose_scale_bar
from semgrains.pipeline import segment_all, METHODS
from semgrains.export import export_result, zip_folder, overlay, stage_arrays, STAGE_NAMES, safe_stem
from semgrains.editing import apply_edit, replay_edits
from semgrains.measurements import measure_grains
from semgrains.ui_components import editor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "interactive"
OUT.mkdir(parents=True, exist_ok=True)
st.set_page_config(page_title="Semgrains", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>.block-container{padding-top:2rem;max-width:1450px}h1{letter-spacing:-.025em}[data-testid=stMetric]{background:#f1f7f7;padding:14px;border-radius:8px}div[data-testid=stSidebar]{background:#f5f8fa}</style>""", unsafe_allow_html=True)
st.title("Semgrains")
st.caption("Сегментация и измерение зёрен на SEM-снимках · Локальная обработка на CPU")


@st.cache_data(max_entries=6, show_spinner=False)
def run_image(data, name, config_dict):
    config = Config(**config_dict)
    return segment_all(load_image(data, name), config)


with st.sidebar:
    st.header("Исходные данные")
    uploads = st.file_uploader("PNG, TIFF, JPEG или BMP", type=["png", "tif", "tiff", "jpg", "jpeg", "bmp"], accept_multiple_files=True)

sources = {}
for uploaded in uploads or []:
    original_name = Path(uploaded.name).name
    upload_name = original_name
    counter = 2
    while upload_name in sources:
        upload_name = f"{Path(original_name).stem}_{counter}{Path(original_name).suffix}"
        counter += 1
    sources[upload_name] = uploaded.getvalue()

if not sources:
    st.info("Загрузите один или несколько снимков в левой панели, чтобы начать работу")
    st.markdown("Поддерживаются PNG, TIFF, JPEG и BMP. После загрузки доступны три метода сегментации, измерения, исправление масок и ручная разметка.")
    st.stop()

with st.sidebar:
    name = st.selectbox("Снимок", list(sources))
    data = sources[name]
    try:
        source = load_image(data, name)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    auto_crop = detect_crop(source.gray)
    st.caption(f"{source.gray.shape[1]} × {source.gray.shape[0]} · {source.mode} · {source.dtype}")
    if source.frames > 1:
        st.warning("Многостраничный TIFF: обрабатывается первый кадр")
    base = Config()
    with st.form("parameters"):
        st.subheader("Область и параметры")
        crop = st.number_input("Первая строка служебной полосы (y)", min_value=8, max_value=source.gray.shape[0], value=auto_crop, key="crop_" + name)
        diameter = st.number_input("Характерный диаметр, px (0 = авто)", min_value=0., max_value=500., value=base.diameter_px, step=1.)
        clip = st.slider("Локальный контраст CLAHE", .5, 4., base.clahe_clip, .1)
        prominence = st.slider("Устойчивость маркеров", .02, .25, base.marker_prominence, .01)
        merge = st.slider("Порог слабой границы для объединения", .0, .40, base.merge_threshold, .01)
        split = st.slider("Порог подтверждения разделения", .30, .90, base.split_threshold, .01)
        adaptive = st.checkbox("Адаптивная постобработка", value=True)
        enhance_boundaries = st.checkbox("Усиление слабых границ", value=base.valley_weight > 0)
        st.subheader("Масштаб")
        st.caption("Подтвердите длину линейки и её подпись. Без подтверждения единицы — px.")
        candidate = propose_scale_bar(source.gray, auto_crop)
        scale_px = st.number_input("Длина линейки, px", min_value=1., value=float(candidate or 10), key="scalepx_"+name)
        scale_val = st.number_input("Значение подписи", min_value=.001, value=200.)
        scale_unit = st.selectbox("Единица подписи", ["nm", "um"])
        confirmed = st.checkbox("Я проверил(а) масштаб по исходному снимку", value=False, key="confirmed_"+name)
        process = st.form_submit_button("Обработать снимок", type="primary", use_container_width=True)
    if process:
        try:
            cfg = Config(crop_bottom=int(crop), diameter_px=diameter, clahe_clip=clip,
                         marker_prominence=prominence, merge_threshold=merge, split_threshold=split,
                         adaptive=adaptive, valley_weight=base.valley_weight if enhance_boundaries else 0.,
                         scale_length_px=scale_px if confirmed else None,
                         scale_length_value=scale_val if confirmed else None, scale_unit=scale_unit)
            st.session_state["active_config"] = cfg.to_dict()
            st.session_state["processed_source"] = sha256(data).hexdigest()
        except ValueError as exc:
            st.error(str(exc))

page = st.radio("Раздел", ["Обработка", "Пакетная обработка", "Ручная разметка"], horizontal=True, label_visibility="collapsed")

if page == "Обработка":
    cfg_dict = st.session_state.get("active_config", base.to_dict())

    if st.session_state.get("processed_source") != sha256(data).hexdigest():
        st.image(source.rgb, caption="Исходный снимок со служебной полосой", width="stretch")
        st.info("Проверьте рабочую область и нажмите «Обработать снимок» в левой панели")
        st.stop()
    with st.spinner("Выделяем зёрна тремя методами…"):
        results = run_image(data, name, cfg_dict)
    method = st.segmented_control("Метод", METHODS, default="ensemble", format_func=lambda x: {"dark":"Тёмные границы", "gradient":"Градиент", "ensemble":"Ансамбль"}[x]) or "ensemble"
    result = results[method]
    token = sha256(data + json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()[:14] + method
    state_key = "edits_" + token
    ops = st.session_state.setdefault(state_key, [])
    labels = replay_edits(result.labels, ops)
    table = measure_grains(labels, source.gray[:result.crop], result.edge, result.config)
    valid = table[table.included]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Полных зёрен", len(valid))
    c2.metric("У края — исключены", len(table)-len(valid))
    c3.metric("Медианный диаметр", f"{valid.equivalent_diameter.median():.1f} {result.config.unit}" if len(valid) else "—")
    c4.metric("Время алгоритма", f"{result.seconds:.2f} с")
    if result.diagnostics["low_contrast"]:
        st.warning("Низкий контраст: проверьте границы вручную")
    tabs = st.tabs(["Результат и исправления", "Все стадии", "Измерения", "Сохранение"])
    with tabs[0]:
        st.caption("Зелёные границы — полные объекты; оранжевые — объекты у края, исключённые из статистики")
        action = editor(overlay(source.rgb, labels), labels, revision=token+str(len(ops)), key="edit_"+token)
        if action and action.get("nonce") != st.session_state.get("last_action_"+token):
            st.session_state["last_action_"+token] = action["nonce"]
            try:
                apply_edit(labels, action)
                ops.append(action)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        left, right = st.columns(2)
        if left.button("Отменить последнее исправление", disabled=not ops):
            ops.pop()
            st.rerun()
        if right.button("Вернуть автоматический результат", disabled=not ops):
            ops.clear()
            st.rerun()
        st.caption(f"Исправлений: {len(ops)}. Карта спорных границ доступна на вкладке стадий.")
    with tabs[1]:
        for start in (0, 3):
            cols = st.columns(3)
            for col, title, a in zip(cols, STAGE_NAMES[start:start+3], stage_arrays(result)[start:start+3]):
                col.image(a, caption=title, width="stretch")
        st.image(np.uint8(result.disagreement)*255, caption="Расхождения границ двух методов (белое — спорные участки)", width="stretch")
        st.json(result.diagnostics, expanded=False)
    with tabs[2]:
        st.dataframe(valid, hide_index=True, width="stretch")
        if len(valid):
            counts, bins = np.histogram(valid.equivalent_diameter, bins=15)
            histogram = pd.DataFrame({"Диаметр": [f"{a:.1f}–{b:.1f}" for a,b in zip(bins[:-1],bins[1:])], "Количество": counts})
            st.bar_chart(histogram, x="Диаметр", y="Количество", x_label="Диаметр, " + result.config.unit, sort=False)
        st.caption("Координаты центров даны в пикселях исходного снимка. Однородность границы = 1 / (1 + CV градиента); это диагностический признак, не вероятность правильности.")
    with tabs[3]:
        revision_hash = sha256(json.dumps(ops, sort_keys=True).encode()).hexdigest()[:10]
        archive_key = "zip_"+token+"_"+revision_hash
        if st.button("Подготовить архив результатов", type="primary"):
            folder = OUT / (safe_stem(name)+"_"+token+"_"+revision_hash)
            for m, r in results.items():
                export_result(r, folder / "automatic" / m)
            if ops:
                export_result(result, folder / "corrected" / method, labels, ops)
            st.session_state[archive_key] = zip_folder(folder)
            st.success(f"Сохранено: {folder}")
        if archive_key in st.session_state:
            st.download_button("Скачать ZIP", st.session_state[archive_key], safe_stem(name)+"_results.zip", "application/zip")
        corrections = {"source": name, "source_sha256": sha256(data).hexdigest(), "method": method,
                       "config": cfg_dict, "operations": ops}
        st.download_button("Сохранить исправления JSON", json.dumps(corrections, ensure_ascii=False, indent=2), "corrections.json", "application/json")
        edit_file = st.file_uploader("Восстановить исправления из JSON", type=["json"])
        if edit_file and st.button("Загрузить исправления"):
            try:
                payload = json.loads(edit_file.getvalue())
                if payload["source_sha256"] != sha256(data).hexdigest() or payload["method"] != method or payload["config"] != cfg_dict:
                    raise ValueError("Исправления относятся к другому снимку, методу или параметрам")
                replay_edits(result.labels, payload["operations"])
                st.session_state[state_key] = payload["operations"]
                st.rerun()
            except (ValueError, KeyError, TypeError) as exc:
                st.error(str(exc))

elif page == "Пакетная обработка":
    st.subheader("Обработка набора с едиными параметрами")
    selected = st.multiselect("Снимки", list(sources), default=list(sources))
    st.caption("Рабочая область определяется для каждого снимка автоматически. Масштаб отдельного снимка не переносится на остальные.")
    batch_cfg = replace(Config(**st.session_state.get("active_config", base.to_dict())), crop_bottom=None, scale_length_px=None, scale_length_value=None)
    if st.button("Обработать выбранные", type="primary", disabled=not selected):
        digest = sha256(json.dumps({"config":batch_cfg.to_dict(), "sources":{n:sha256(sources[n]).hexdigest() for n in selected}}, sort_keys=True).encode()).hexdigest()[:12]
        folder = OUT / ("batch_"+digest)
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        bar = st.progress(0)
        for i, filename in enumerate(selected):
            try:
                results = run_image(sources[filename], filename, batch_cfg.to_dict())
                for m, r in results.items():
                    meta = export_result(r, folder / safe_stem(filename) / m)
                    rows.append({"source": filename, "method": m, "count": meta["count_included"], "seconds": r.seconds, "error": ""})
            except Exception as exc:
                rows.append({"source": filename, "error": str(exc)})
            bar.progress((i+1)/len(selected), text=filename)
        pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False, encoding="utf-8-sig")
        st.session_state["batch_zip"] = zip_folder(folder)
        st.session_state["batch_table"] = rows
    if "batch_zip" in st.session_state:
        st.dataframe(st.session_state["batch_table"], hide_index=True)
        st.download_button("Скачать результаты набора", st.session_state["batch_zip"], "batch_results.zip", "application/zip")

elif page == "Ручная разметка":
    st.subheader("Независимая разметка контрольного участка")
    st.caption("Эталон рисуется по исходному снимку без автоматических масок. Неоднозначные области отмечайте отдельно.")
    h, w = source.gray.shape
    with st.form("patch"):
        c1, c2, c3, c4 = st.columns(4)
        px = c1.number_input("x начала", 0, w-16, max(0, w//2-48))
        py = c2.number_input("y начала", 0, auto_crop-16, max(0, auto_crop//2-48))
        pw = c3.number_input("Ширина", 16, w, min(96,w))
        ph = c4.number_input("Высота", 16, auto_crop, min(96,auto_crop))
        if st.form_submit_button("Открыть участок"):
            if px+pw > w or py+ph > auto_crop:
                st.error("Участок выходит за рабочую область")
            else:
                st.session_state["annotation_patch"] = {"source": name, "box": [int(px),int(py),int(pw),int(ph)]}
    patch = st.session_state.get("annotation_patch")
    if patch and patch["source"] == name:
        x,y,pw,ph = patch["box"]
        ptoken = sha256((name+str(patch["box"])).encode()).hexdigest()[:12]
        answer = editor(source.rgb[y:y+ph,x:x+pw], key="annotation_"+ptoken, revision=ptoken)
        reviewed = st.checkbox("Я визуально проверил(а) все контуры и неопределённые области")
        reviewer = st.text_input("Автор / проверяющий", value="")
        if answer and st.button("Сохранить разметку", disabled=not reviewed or not reviewer.strip()):
            if not answer.get("polygons"):
                st.error("Нет контуров зёрен")
            else:
                payload = {**patch, "polygons": answer["polygons"], "ignored": answer.get("ignored", []),
                           "reviewed": True, "reviewer": reviewer.strip(), "provenance": "independent_manual_trace",
                           "source_sha256": sha256(data).hexdigest()}
                folder = ROOT / "annotations" / "user"
                folder.mkdir(parents=True, exist_ok=True)
                (folder/(ptoken+".json")).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                st.success("Разметка сохранена отдельно от автоматических результатов")
