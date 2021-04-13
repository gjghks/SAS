from data import COLORS
from sewer import Sewer
from utils.augmentations import FastBaseTransform
from utils.functions import MovingAverage, ProgressBar
from utils import timer
from layers.output_utils import postprocess, undo_image_transformation

from data import cfg
from data.config import Config

import torch
import torch.backends.cudnn as cudnn
import argparse
import time
import random
import os
from collections import defaultdict
from pathlib import Path
from PIL import Image

import cv2
import pyshine as ps
import csv
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
custom_config = '--oem 3 --psm 6 outputbase digits tessedit_char_whitelist=0123456789'
# custom_config = '--oem 1 --psm 8 -c tessedit_char_whitelist=.0123456789'
# custom_config = '--oem 3 --psm 13 -c tessedit_char_whitelist=.0123456789'

from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet
from generate_html import generate_html

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.jinja_env.auto_reload = True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='SEWER COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/sewer_im700_311_697008.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--config', default='sewer_im700_config',
                        help='The config object to use.')
    parser.add_argument('--seed', default=0, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does '
                             'not (I think) affect cuda stuff.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the '
                             'format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_fps', default=30, type=int,
                        help='The number of frames to analysis video according to fps')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0.50, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently '
                             'only works in display mode.')

    parser.set_defaults(fast_nms=True, cross_class_nms=False, no_bar=False, display=False, resume=False, max_images=-1,
                        output_coco_json=False, output_web_json=False,
                        shuffle=False, display_lincomb=False, dataset=None,
                        benchmark=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False,
                        display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)


color_cache = defaultdict(lambda: {})

result_list = []
frame_compare = -1


def ocr(img):
    img_copy = img.copy()
    image_roi = img_copy[15:60, 40:143]

    image_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    image_roi = cv2.threshold(image_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # cv2.imshow('test', image_roi)
    # cv2.waitKey(0)

    image_roi = Image.fromarray(image_roi)
    dist = pytesseract.image_to_string(image_roi, config=custom_config)
    dist = dist.replace('\n\f', '')

    if len(dist) == 4 and dist.find('.') == -1:
        dist = dist[:-1] + '.' + dist[-1:]

    return dist


def prep_display_for_img(dets_out, img, h=None, w=None, undo_transform=True, class_color=False, mask_alpha=0.45):
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                        crop_masks=args.crop,
                        score_threshold=args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]

        if cfg.eval_mask_branch:
            masks = t[3][idx]
        classes, scores, boxes = [x[idx] for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] if class_color else j) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        masks = masks[:num_dets_to_consider, :, :, None]

        colors = torch.cat(
            [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        inv_alph_masks = masks * (-mask_alpha) + 1

        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if num_dets_to_consider == 0:
        return img_numpy

    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    font_thickness = 1

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                if args.display_scores:
                    text_str_class = f"{_class}"
                    text_str_score = f": {score:.2f}"

                    text_w_class, text_h_class = cv2.getTextSize(text_str_class, font_face, font_scale, font_thickness)[0]

                    img_numpy = ps.putBText(img_numpy, text_str_class, text_offset_x=x1, text_offset_y=y1,
                                            vspace=0, hspace=0, font=font_face, font_scale=0.6,
                                            thickness=font_thickness,
                                            alpha=0.7, background_RGB=color, text_RGB=(255, 255, 255))
                    img_numpy = ps.putBText(img_numpy, text_str_score, text_offset_x=x1,
                                            text_offset_y=y1 + text_h_class + 2,
                                            vspace=0, hspace=0, font=font_face, font_scale=0.6,
                                            thickness=font_thickness,
                                            alpha=0.7, background_RGB=color, text_RGB=(255, 255, 255))
                else:
                    text_str_class = '%s' % _class

                    img_numpy = ps.putBText(img_numpy, text_str_class, text_offset_x=x1, text_offset_y=y1,
                                            vspace=0, hspace=0, font=font_face, font_scale=0.6,
                                            thickness=font_thickness,
                                            alpha=0.7, background_RGB=color, text_RGB=(255, 255, 255))

    return img_numpy


def prep_display_for_video(dets_out, img, h=None, w=None, save_folder=None, undo_transform=True, class_color=False,
                 mask_alpha=0.45, fps_str='', override_args: Config = None):
    if undo_transform:
        assert w is not None and h is not None, "with undo_transform=True, w,h params must be specified!"
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    img_numpy_ori = (img_gpu * 255).byte().cpu().numpy()

    global args
    if override_args is not None:
        args = override_args

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                        crop_masks=args.crop,
                        score_threshold=args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]

        if cfg.eval_mask_branch:
            masks = t[3][idx]
        classes, scores, boxes = [x[idx] for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] if class_color else j) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    global frame_compare

    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        if frame_compare != save_folder[4]:
            masks = masks[:num_dets_to_consider, :, :, None]

            colors = torch.cat(
                [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)],
                dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

            inv_alph_masks = masks * (-mask_alpha) + 1

            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if num_dets_to_consider == 0:
        if os.path.isdir(save_folder[0]) and save_folder[4] % args.video_fps == 0:
            file_name = save_folder[1] + "_%05d" % save_folder[4] + '.png'
            cv2.imwrite(os.path.join(save_folder[3], file_name), img_numpy)
            cv2.imwrite(os.path.join(save_folder[2], file_name), img_numpy_ori)

        return [img_numpy, img_numpy_ori]

    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    font_thickness = 1

    if args.display_text or args.display_bboxes:
        if frame_compare != save_folder[4]:
            frame_compare = save_folder[4]
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]

                if args.display_bboxes:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                if args.display_text:
                    _class = cfg.dataset.class_names[classes[j]]
                    # text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class
                    if args.display_scores:
                        text_str_class = f"{_class}"
                        text_str_score = f": {score:.2f}"

                        text_w_class, text_h_class = \
                            cv2.getTextSize(text_str_class, font_face, font_scale, font_thickness)[0]

                        img_numpy = ps.putBText(img_numpy, text_str_class, text_offset_x=x1, text_offset_y=y1,
                                                vspace=0, hspace=0, font=font_face, font_scale=0.6,
                                                thickness=font_thickness,
                                                alpha=0.7, background_RGB=color, text_RGB=(255, 255, 255))
                        img_numpy = ps.putBText(img_numpy, text_str_score, text_offset_x=x1,
                                                text_offset_y=y1 + text_h_class + 2,
                                                vspace=0, hspace=0, font=font_face, font_scale=0.6,
                                                thickness=font_thickness,
                                                alpha=0.7, background_RGB=color, text_RGB=(255, 255, 255))
                    else:
                        text_str_class = '%s' % (_class)

                        img_numpy = ps.putBText(img_numpy, text_str_class, text_offset_x=x1, text_offset_y=y1,
                                                vspace=0, hspace=0, font=font_face, font_scale=0.6,
                                                thickness=font_thickness,
                                                alpha=0.7, background_RGB=color, text_RGB=(255, 255, 255))

                    if save_folder[4] % args.video_fps == 0:
                        dist = ocr(img_numpy_ori)
                        result = save_folder[
                                     4], f"{dist}", f"{_class}", f"{score:.2f}", f"{x1}", f"{y1}", f"{x2}", f"{y2}"
                        result_list.append(result)

            if os.path.isdir(save_folder[0]) and save_folder[4] % args.video_fps == 0:
                file_name = save_folder[1] + "_%05d" % save_folder[4] + '.png'
                cv2.imwrite(os.path.join(save_folder[3], file_name), img_numpy)
                cv2.imwrite(os.path.join(save_folder[2], file_name), img_numpy_ori)

            return [img_numpy, img_numpy_ori, result_list]

    return [img_numpy, img_numpy_ori]


def evalimage(net: Sewer, path: str, save_path: str = None):
    with torch.no_grad():
        frame = torch.from_numpy(cv2.imread(path)).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)

        img_numpy = prep_display_for_img(preds, frame, undo_transform=False, class_color=True)

        if save_path is not None:
            cv2.imwrite(save_path, img_numpy)

        return img_numpy


def evalimages(net: Sewer, input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print()
    for p in Path(input_folder).glob('*'):
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.png'
        out_path = os.path.join(output_folder, name)

        evalimage(net, path, out_path)
        print(path + ' -> ' + out_path)
    print('Done.')


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


from multiprocessing.pool import ThreadPool
from queue import Queue


class CustomDataParallel(torch.nn.DataParallel):

    def gather(self, outputs, output_device):
        return sum(outputs, [])


def file_csv(name, data):
    with open(name, 'w', newline='') as out:
        csv_out = csv.writer(out)
        for row in data:
            csv_out.writerow(row)


def evalvideo(net: Sewer, path: str, out_path: str = None):
    is_webcam = path.isdigit()

    cudnn.benchmark = True

    if is_webcam:
        vid = cv2.VideoCapture(int(path))
    else:
        vid = cv2.VideoCapture(path)

    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)

    target_fps = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if is_webcam:
        num_frames = float('inf')
    else:
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(100)
    fps = 0
    frame_time_target = 1 / target_fps
    running = True
    fps_str = ''
    vid_done = False
    frames_displayed = 0
    out_images_path = path.split('.')[0]

    base_name = os.path.splitext(os.path.basename(path))[0]
    createFolder(out_images_path)
    ori_folder = os.path.join(out_images_path + '\#ori')
    res_folder = os.path.join(out_images_path + '\#res')

    createFolder(ori_folder)
    createFolder(res_folder)

    if out_path is not os.path.isdir(out_path):
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), target_fps, (frame_width, frame_height))

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        if out_path is not None:
            out.release()
        cv2.destroyAllWindows()
        #exit()

    def get_next_frame(vid):
        frames = []
        for idx in range(args.video_multiframe):
            frame = vid.read()[1]
            if frame is None:
                return frames
            frames.append(frame)
        return frames

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp
            num_extra = 0
            while imgs.size(0) < args.video_multiframe:
                imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
                num_extra += 1
            out = net(imgs)
            if num_extra > 0:
                out = out[:-num_extra]
            return frames, out

    def prep_frame(inp, fps_str, save_folder=None):
        with torch.no_grad():
            frame, preds = inp
            return prep_display_for_video(preds, frame, save_folder=save_folder, undo_transform=False, class_color=True,
                                fps_str=fps_str, override_args=args)

    frame_buffer = Queue()
    video_fps = 0

    def play_video():
        try:
            nonlocal frame_buffer, running, video_fps, is_webcam, num_frames, frames_displayed, vid_done

            video_frame_times = MovingAverage(100)
            frame_time_stabilizer = frame_time_target
            last_time = None
            stabilizer_step = 0.0005
            progress_bar = ProgressBar(30, num_frames)

            while running:
                frame_time_start = time.time()

                if not frame_buffer.empty():
                    next_time = time.time()
                    if last_time is not None:
                        video_frame_times.add(next_time - last_time)
                        video_fps = 1 / video_frame_times.get_avg()
                    if out_path is None or os.path.isdir(out_path):
                        cv2.imshow(path, frame_buffer.get()[0])
                    else:
                        out.write(frame_buffer.get()[0])
                    frames_displayed += 1
                    last_time = next_time

                    if out_path is not None and not os.path.isdir(out_path):
                        if video_frame_times.get_avg() == 0:
                            fps = 0
                        else:
                            fps = 1 / video_frame_times.get_avg()
                        progress = frames_displayed / num_frames * 100
                        progress_bar.set_val(frames_displayed)
                        print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                              % (repr(progress_bar), frames_displayed, num_frames, progress, fps), end='')

                if (out_path is None or os.path.isdir(out_path)) and cv2.waitKey(1) == 27:
                    running = False

                if not vid_done:
                    buffer_size = frame_buffer.qsize()
                    if buffer_size < args.video_multiframe:
                        frame_time_stabilizer += stabilizer_step
                    elif buffer_size > args.video_multiframe:
                        frame_time_stabilizer -= stabilizer_step
                        if frame_time_stabilizer < 0:
                            frame_time_stabilizer = 0

                    new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)
                else:
                    new_target = frame_time_target

                next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
                target_time = frame_time_start + next_frame_target - 0.001

                if out_path is None or os.path.isdir(out_path) or args.emulate_playback:
                    while time.time() < target_time:
                        time.sleep(0.001)
                else:
                    time.sleep(0.001)
        except:
            import traceback
            traceback.print_exc()

    extract_frame = lambda x, i: (
        x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])

    print('Initializing model... ', end='')
    first_batch = eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    sequence = [prep_frame, eval_network, transform_frame]
    pool = ThreadPool(processes=len(sequence) + args.video_multiframe + 2)
    pool.apply_async(play_video)
    active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in range(len(first_batch[0]))]

    print()
    if out_path is None or os.path.isdir(out_path): print('Press Escape to close.')
    try:
        while vid.isOpened() and running:
            while frame_buffer.qsize() > 100:
                time.sleep(0.001)

            start_time = time.time()

            if not vid_done:
                next_frames = pool.apply_async(get_next_frame, args=(vid,))
            else:
                next_frames = None

            if not (vid_done and len(active_frames) == 0):
                for frame in active_frames:
                    _args = [frame['value']]
                    if frame['idx'] == 0:
                        _args.append(fps_str)
                        if out_path is not None:
                            _args.append([out_images_path, base_name, ori_folder, res_folder, frames_displayed])
                    frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)

                for frame in active_frames:
                    if frame['idx'] == 0:
                        frame_buffer.put(frame['value'].get())

                active_frames = [x for x in active_frames if x['idx'] > 0]

                for frame in list(reversed(active_frames)):
                    frame['value'] = frame['value'].get()
                    frame['idx'] -= 1

                    if frame['idx'] == 0:
                        active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in
                                          range(1, len(frame['value'][0]))]
                        frame['value'] = extract_frame(frame['value'], 0)

                if next_frames is not None:
                    frames = next_frames.get()
                    if len(frames) == 0:
                        vid_done = True
                    else:
                        active_frames.append({'value': frames, 'idx': len(sequence) - 1})

                frame_times.add(time.time() - start_time)
                if frame_times.get_avg() != 0:
                    fps = args.video_multiframe / frame_times.get_avg()
            else:
                fps = 0
                running = False

            fps_str = 'Processing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d' % (
                fps, video_fps, frame_buffer.qsize())
            if not args.display_fps and os.path.isdir(out_path):
                print('\r' + fps_str + '    ', end='')

    except KeyboardInterrupt:
        print('\nStopping...')
        if os.path.isdir(out_images_path):
            file_csv(os.path.join(out_images_path, base_name + '.csv'), result_list)

    if os.path.isdir(out_images_path):
        file_csv(os.path.join(out_images_path, base_name + '.csv'), result_list)

    cleanup_and_exit()


def evaluate(net: Sewer):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            evalimage(net, inp, out)
            generate_html(inp, out)
        return
    elif args.images is not None:
        for imgList in args.images:
            inp, out = imgList.split(':')
            evalimage(net, inp, out)
        generate_html(img_list=args.images)
        return
    elif args.video is not None:
        if ':' in args.video:
            inp, out = args.video.split(':')
            evalvideo(net, inp, out)
            generate_html(inp, out)
            return
        else:
            evalvideo(net, args.video)
            return


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from werkzeug.utils import secure_filename


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        # file = request.files.get('file')
        files = request.files.getlist("file")
        if not files:
            return

        imagelist = []
        allow_ext = ['jpg', 'jpeg', 'png']

        for value in files:
            if value.content_type.split('/')[0] == 'image':
                if value.filename.split('.')[1] != allow_ext[0] and value.filename.split('.')[1] != allow_ext[1] and \
                        value.filename.split('.')[1] != allow_ext[2]:
                #if value.filename.split('.')[1] != allow_ext[0] and value.filename.split('.')[1] != 'png':
                    print("Files are not Image")
                    return render_template('index.html')
                else:
                    filename = secure_filename(value.filename)
                    value.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    imagelist.append('static/' + filename + ":static/" + filename.split('.')[0] + "_result.png")
                    args.images = imagelist
                    args.image = None
                    args.video = None
            elif value.content_type.split('/')[0] == 'video':
                videoname = secure_filename(value.filename)
                value.save(os.path.join(app.config['UPLOAD_FOLDER'], videoname))
                args.video = 'static/' + videoname + ":static/" + videoname.split('.')[0] + "_result.mp4"
                args.images = None
                args.image = None

        evaluate(net)

        return redirect(url_for('success', name='results'))


@app.route('/success/<name>')
def success(name):
    return render_template('results.html')


if __name__ == '__main__':

    parse_args()

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        dataset = None

        print('Loading model...', end='')
        net = Sewer()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        app.run(debug=True, use_reloader=False)