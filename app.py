from __future__ import annotations

import argparse
from dataclasses import fields
import json
import logging
from pathlib import Path
import sys

import gradio as gr

from handdraw.settings import INK_DEFAULTS, RIG_SETTINGS, Settings

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'outputs'
PROTECTED_RIG_KEYS = {'reference_height', 'native_scale', 'wrist_degrees', 'wrist_frequency'}
LOGGER = logging.getLogger(__name__)


def render_image(image, values, progress=None):
    try:
        if image is None:
            raise ValueError('Upload an input image.')
        settings = Settings(**values).validate()
        from handdraw.render import render_video

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        callback = None if progress is None else lambda fraction, description: progress(fraction, desc=description)
        output, report = render_video(image, settings, output_dir=OUTPUT_DIR, progress=callback)
        output = Path(output).resolve()
        if not output.is_relative_to(OUTPUT_DIR.resolve()) or output.suffix.lower() != '.mp4' or not output.is_file():
            raise ValueError('Renderer returned an invalid output file.')
        LOGGER.info('Rendered %s', output.name)
        return str(output), str(output), report
    except Exception as exc:
        LOGGER.exception('Render failed')
        raise gr.Error(str(exc)) from exc


def setting_control(item):
    spec = item.metadata
    options = dict(label=spec['label'], value=item.default)
    if spec['choices']:
        return gr.Radio(choices=spec['choices'], **options) if item.name == 'mode' else gr.Dropdown(choices=spec['choices'], **options)
    if isinstance(item.default, bool):
        return gr.Checkbox(**options)
    if spec['low'] is not None:
        options.update(minimum=spec['low'], maximum=spec['high'], step=spec['step'])
        if item.name == 'seed':
            return gr.Number(precision=0, **options)
        return gr.Slider(**options)
    if item.name == 'paper_color':
        return gr.ColorPicker(**options)
    if item.name in ('rig_json', 'ink_json'):
        defaults = ({key: value for key, value in RIG_SETTINGS.items() if key not in PROTECTED_RIG_KEYS}
                    if item.name == 'rig_json' else INK_DEFAULTS)
        options['value'] = json.dumps(defaults, indent=2)
        return gr.Textbox(lines=12, max_lines=24, **options)
    return gr.Textbox(**options)


def build_ui():
    with gr.Blocks(title='Handdraw', analytics_enabled=False) as demo:
        controls = {}
        items_by_name = {item.name: item for item in fields(Settings)}
        gr.Markdown('## Handdraw\nCreate hand-drawn animations from your images.', elem_id='studio-header')
        with gr.Row(equal_height=True, elem_id='media-row'):
            with gr.Column(scale=1, min_width=320):
                image = gr.Image(label='Original image', type='numpy', image_mode='RGBA', sources=['upload'], height=360)
            with gr.Column(scale=1, min_width=320):
                video = gr.Video(label='Rendered video', format='mp4', interactive=False, height=360)
        with gr.Group(elem_id='quick-settings'):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=300):
                    controls['mode'] = setting_control(items_by_name['mode'])
                    gr.Markdown('Both modes prioritize characters. **Quick** colors background clusters from left to right.', elem_classes='hint')
                with gr.Column(scale=2, min_width=300):
                    with gr.Row():
                        for name in ('duration', 'fps'):
                            controls[name] = setting_control(items_by_name[name])
                    with gr.Row():
                        for name in ('show_hand', 'subject_first'):
                            controls[name] = setting_control(items_by_name[name])
        with gr.Row(elem_id='export-row'):
            submit = gr.Button('Render & export MP4', variant='primary', size='lg', scale=1, min_width=240)
            download = gr.File(label='Download MP4', interactive=False, type='filepath', scale=1, min_width=240)
        group_labels = {'Video': 'Frame & timing', 'Advanced': 'Advanced settings (JSON)'}
        with gr.Row(elem_id='advanced-settings'):
            for groups in (('Video', 'Lines & colors', 'Coloring path'), ('Hand & pen', 'Video export', 'Advanced')):
                with gr.Column(scale=1, min_width=320):
                    for group in groups:
                        with gr.Accordion(group_labels.get(group, group), open=False):
                            if group == 'Video export':
                                gr.Markdown('No NVIDIA GPU? Select **CPU** and **CPU H.264**.', elem_classes='hint')
                            items = [item for item in fields(Settings)
                                     if item.metadata['group'] == group and item.name not in controls]
                            for first in range(0, len(items), 2):
                                with gr.Row():
                                    for item in items[first:first + 2]:
                                        controls[item.name] = setting_control(item)
        with gr.Accordion('Render details', open=False):
            report = gr.JSON(label='Report')

        def generate(data, progress=gr.Progress()):
            return render_image(data[image], {name: data[control] for name, control in controls.items()}, progress)

        submit.click(generate, inputs={image, *controls.values()}, outputs=[video, download, report],
                     concurrency_limit=1, api_name='render')
    demo.setting_controls = controls
    demo.queue(default_concurrency_limit=1, max_size=8)
    return demo


def main():
    parser = argparse.ArgumentParser(description='Local Handdraw UI')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--open', action='store_true', dest='open_browser')
    parser.add_argument('--log', nargs='?', const=str(ROOT / 'logs' / 'app.log'), metavar='PATH')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be 1..65535')
    if args.log:
        path = Path(args.log).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open('a', encoding='utf-8', buffering=1)
        sys.stdout = sys.stderr = stream
        logging.basicConfig(level=logging.INFO, stream=stream,
                            format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_ui().launch(server_name='127.0.0.1', server_port=args.port, share=False,
                      inbrowser=args.open_browser, allowed_paths=[str(OUTPUT_DIR.resolve())],
                      blocked_paths=[str(ROOT / '.venv'), str(ROOT / 'logs'),
                                     str(ROOT / 'assets' / 'v181d' / 'model.safetensors')],
                      max_file_size='20mb', footer_links=[],
                      theme=gr.themes.Default(primary_hue='orange', neutral_hue='slate', font=['Segoe UI', 'sans-serif']),
                      css_paths=ROOT / 'assets' / 'studio.css')


if __name__ == '__main__':
    main()
