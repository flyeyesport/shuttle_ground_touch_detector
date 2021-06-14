import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import glob
from math import sqrt
import xml.etree.ElementTree as xml
import argparse
import random


class ShuttleClipsInSequencesDataset(Dataset):

    def __init__(self, root_dir, use_only_with_touch=False,
                 use_only_without_touch=False, use_all=False,
                 generate_sequences=False, randomize_shake=False,
                 shaken_frames_count=200, extra_previous_frames_count=0,
                 ground_truth_as_index=True, ground_truth_shift=0,
                 extended_patch_size=136, patch_size=128, frames_count=16,
                 only_first_touches=False, with_patch_positions=False,
                 original_image_height=600, original_image_width=800,
                 normalize_patch_positions=False, randomize_shake_shift_all=1):
        """Constructor.

        It initializes the parameters that change the way the dataset is
        constructed and the data is returned.
        The iterator can return a set of N consecutive frames from one
        sequence, or it can return the whole sequence, or it can return an
        extended set of K + N frames, where K is a number of prepended frames.
        The ground truth that is returned together with the data can also
        be returned in different ways: as a number indicating the index of the
        frame with ground touch or a list of N or K+N flags with values of 0
        or 1. There are even more options. See the descriptions of the
        parameters.

        Args:
            root_dir (str): Path to the directory with subdirectories with data

            use_only_with_touch (bool): Boolean flag indicating if only sets
            containing frame where shuttle touched ground should be returned.
            use_only_without_touch (bool): Boolean flag indicating if only sets
            that don't contain frame where shuttle touched ground should be
            returned.

            use_all (bool): Boolean flag indicating if all sets (those
            containing and those not containing shuttle ground touches) should
            be returned. If all three flags: use_only_with_touch,
            use_only_without_touch and use_all are set to False, then all data
            is returned, as though the use_all flag would have a value set to
            True.

            generate_sequences (bool):
            IMPORTANT: first fix __get_item_ids_for_sequence() method to use
            this option.

            If this boolean flag is set to True,
            then the returned result is a list of sets of N frames with
            corresponding ground truth. If it is set to false, then only one
            set of N frames together with ground truth is returned.

            randomize_shake (bool): The frames are squares of 136x136 pixels,
            but only squares of 128x128 pixels are returned by iterator. If
            this flag is set to False then from the frames of 136x136, 9x9 sets
            of 128x128 pixels are generated. All frames in the set are shifted
            the same from original frames of 136x136. If this flag is set to
            True then each frame of 128x128 is randomly cropped from 136x136
            frames, so the resulted set has "shaking" frames. See also:
            randomize_shake_shift_all flag which allows to shift all controls

            shaken_frames_count (int): one frame of 128x128 pixels can be
            cropped from a 136x136 pixels frame in 9*9 ways. This is ok, if we
            crop all frames in the set in the same way (see: randomize_shake),
            but if we crop each frame in the set with different shift, then we
            get a very big number of possible sets. This parameter is a
            number of such sets to generate, instead of all combinations.

            randomize_shake_shift_all (int): if this value is bigger than 0
            then the whole set is shifted in the range [-N,N] where N is the
            value of randomize_shake_shift_all parameter, and then each frame
            is randomly shifted but still to fit 128x128 frame inside a
            136x136 source frame.

            extra_previous_frames_count (int): number of consecutive frames
            (K in the description above) from the same sequence to prepend to
            the returned set of frames. instead of returning N frames we return
            K+N frames. It is useful if we want to use the data to train
            recurrent nets. This prepended subset can be used to feed the net
            before feeding it with the frames for which we have ground truth.

            ground_truth_as_index (bool): If set to true, then the ground true
            is represented as a single value of the index of the frame in the
            returned set in which the shuttle touched ground. If set to false,
            a list of 0 (not ground touch) and 1 (ground touch) is returned.
            The length of the list is:
            frames_count + extra_previous_frames_count - ground_truth_shift.

            ground_truth_shift (int): Number of frames to shift the index of
            the returned ground truth. For example if we want to return 16
            frames with another extra 9 prepended frames, then we could set
            this parameter to 4 to shift the ground truth to the center of the
            prepended frames.

            extended_patch_size (int): Width and height of the image patch in
            which a smaller window is "shaken". See parameter: randomize_shake.

            patch_size (int): Width and height of the frame.

            frames_count (int) Number of consequtive frames returned in a
            frameset.

            only_first_touches (bool): Boolean flag indicating if dataset
            should return only fragments of sequences with first ground
            touches.

            with_patch_positions (bool): Boolean flag indicating if dataset
            should return also positions of each pach in relation to the
            original image from which the patch was taken. The position is
            represented as two numbers which are a position of the left-top
            corner of the path in the original image.

            original_image_height (int): Height of the original frames form
            which the patches are cut. Default: 600.

            original_image_width (int): Width of the original frames form
            which the patches are cut. Default: 800.

            normalize_patch_positions (bool): Boolean flag indicating if
            patch positions should be normalized before returning. Normalized
            patch_positions are float numbers in range from 0 to 1.
    """

        self.generate_sequences = generate_sequences
        self.randomize_shake = randomize_shake
        self.shaken_frames_count = shaken_frames_count
        self.extra_previous_frames_count = extra_previous_frames_count
        self.ground_truth_as_index = ground_truth_as_index
        self.ground_truth_shift = ground_truth_shift
        self.set_len = extra_previous_frames_count + frames_count
        self.frames_count = frames_count
        self.patch_size = patch_size
        self.extended_patch_size = extended_patch_size
        self.only_first_touches = only_first_touches
        self.with_patch_positions = with_patch_positions
        self.original_image_height = original_image_height
        self.original_image_width = original_image_width
        self.normalize_patch_positions = normalize_patch_positions
        d = extended_patch_size - patch_size / 2 + 1
        if randomize_shake_shift_all < 1:
            randomize_shake_shift_all = 1
        elif randomize_shake_shift_all > d:
            randomize_shake_shift_all = d
        self.randomize_shake_shift_all = randomize_shake_shift_all

        if self.randomize_shake:
            random.seed(12345678)
        if (
            not use_only_with_touch and
            not use_only_without_touch and
            not use_all
        ):
            self.use_all = True
            self.use_only_with_touch = False
            self.use_only_without_touch = False
        elif use_only_with_touch:
            self.use_all = False
            self.use_only_with_touch = True
            self.use_only_without_touch = False
        elif use_only_without_touch:
            self.use_all = False
            self.use_only_with_touch = False
            self.use_only_without_touch = True
        else:
            self.use_all = True
            self.use_only_with_touch = False
            self.use_only_without_touch = False

        self.sets_with_touch_count = 0
        self.sets_without_touch_count = 0
        self.sets_count = 0
        self.root_dir = root_dir
        self.counters = []
        if os.path.isdir(root_dir):
            for data_dir in sorted(os.listdir(root_dir)):
                data_dir = os.path.join(root_dir, data_dir)
                if os.path.isdir(data_dir):
                    data_file = os.path.join(data_dir, 'info.xml')
                    if os.path.isfile(data_file):
                        # print(data_file)
                        self.__register_new_directory(data_dir, data_file)
        if self.use_only_without_touch:
            self.actual_sets_count = self.sets_without_touch_count
        elif self.use_only_with_touch:
            self.actual_sets_count = self.sets_with_touch_count
        else:
            self.actual_sets_count = self.sets_count

        if self.generate_sequences:
            self.sequences = []
            for counter_id, counter in enumerate(self.counters):
                if self.use_only_without_touch:
                    list = counter["without_touch"]
                elif self.use_only_with_touch:
                    list = counter["with_touch"]
                else:
                    list = counter["with_touch"] + counter["without_touch"]
                list.sort(key=lambda elem: elem[0])
                previous = None
                first_frame = None
                last_frame = None
                for range in list:
                    if previous is not None:
                        if previous[1] + 1 == range[0]:
                            last_frame = range[1]
                            previous = range
                        else:
                            sequence = {"counter_id": counter_id,
                                        "first_frame": first_frame,
                                        "last_frame": last_frame}
                            self.sequences.append(sequence)
                            previous = range
                            first_frame = range[0]
                            last_frame = range[1]
                    else:
                        previous = range
                        first_frame = range[0]
                        last_frame = range[1]
                if first_frame is not None:
                    sequence = {"counter_id": counter_id,
                                "first_frame": first_frame,
                                "last_frame": last_frame}
                    self.sequences.append(sequence)

        # print(self.sets_without_touch_count, self.sets_with_touch_count)
        # print(self.counters)
        # try to read mean and std deviation from file, if there is no file,
        # calculate them
        if not self.__read_mean_and_std():
            self.__calculate_mean_and_std()
            self.__save_mean_and_std()

    def __register_new_directory(self, data_dir, data_file):
        set = {'with_touch': [], 'without_touch': [], 'all': [],
               'all_prev_count': 0, 'with_touch_prev_count': 0,
               'without_touch_prev_count': 0,
               'all_count': 0, 'with_touch_count': 0,
               'without_touch_count': 0, 'touches': []}
        self.counters.append(set)
        num = len(glob.glob(os.path.join(data_dir, '*.png')))
        if(num >= self.set_len):
            last_counter = self.counters[-1]
            a = last_counter['without_touch']
            b = last_counter['with_touch']
            c = last_counter['all']
            a_count = 0
            b_count = 0
            c_count = 0
            a_start = 0
            try:
                parsed_xml = xml.parse(data_file)
                ground_touches = parsed_xml.findall('ground_touch')
                if self.only_first_touches:
                    node_count = min(len(ground_touches), 1)
                else:
                    node_count = len(ground_touches)
                for node_id in range(node_count):
                    node = ground_touches[node_id]
                    if node is None or node.text is None:
                        break
                    else:
                        touch_idx = int(node.text)
                        last_counter['touches'].append(touch_idx)
                        x = touch_idx - self.set_len
                        if x >= a_start:
                            a.append((a_start, x))
                            a_count += (x - a_start + 1)
                            c.append((a_start, x))
                            c_count += (x - a_start + 1)
                            b_start = x + 1
                        else:
                            b_start = a_start
                        if touch_idx >= num:
                            a_start = num
                        else:
                            b_end = (touch_idx
                                     - self.extra_previous_frames_count)
                            if b_end > num - self.set_len:
                                b_end = num - self.set_len
                            if b_start <= b_end:
                                b.append((b_start, b_end))
                                b_count += b_end - b_start + 1
                                c.append((b_start, b_end))
                                c_count += b_end - b_start + 1
                                a_start = b_end + 1
                            else:
                                a_start = b_start
                if a_start <= num - self.set_len:
                    found_second_touch = False
                    if self.only_first_touches:
                        for node_id in range(1, len(ground_touches)):
                            node = ground_touches[node_id]
                            if node is not None and node.text is not None:
                                found_second_touch = True
                                break
                    if self.only_first_touches and found_second_touch:
                        x = int(ground_touches[node_id].text)
                        if a_start <= x - self.set_len:
                            a.append((a_start, x - self.set_len))
                            a_count += x - self.set_len - a_start + 1
                            c.append((a_start, x - self.set_len))
                            c_count += x - self.set_len - a_start + 1
                    else:
                        a.append((a_start, num - self.set_len))
                        a_count += num - self.set_len - a_start + 1
                        c.append((a_start, num - self.set_len))
                        c_count += num - self.set_len - a_start + 1
                last_counter['without_touch_count'] = a_count
                last_counter['with_touch_count'] = b_count
                last_counter['all_count'] = c_count
                if len(self.counters) >= 2:
                    prev_counter = self.counters[-2]
                    last_counter['without_touch_prev_count'] = (
                        prev_counter['without_touch_count'] +
                        prev_counter['without_touch_prev_count']
                    )
                    last_counter['with_touch_prev_count'] = (
                        prev_counter['with_touch_count'] +
                        prev_counter['with_touch_prev_count']
                    )
                    last_counter['all_prev_count'] = (
                        prev_counter['all_count'] +
                        prev_counter['all_prev_count']
                    )
            except xml.ParseError as err:
                print('ERROR: malformed XML file: ' +
                      data_file + '. Technical info: ' +
                      str(err))
            self.sets_with_touch_count += last_counter['with_touch_count']
            self.sets_without_touch_count += (
                last_counter['without_touch_count'])
            self.sets_count += last_counter['all_count']

    def __read_mean_and_std(self):
        try:
            file_name = os.path.join(self.root_dir, 'mean_and_std.json')
            with open(file_name) as json_file:
                data = json.load(json_file)
                self.mean = float(data['mean'])
                self.std = float(data['std'])
            return True
        except Exception:
            return False

    def __calculate_mean_and_std(self):
        m = 0
        s = 0
        count = 0
        dirs = os.listdir(self.root_dir)
        num = len(dirs)
        i_num = 0
        str = "Calculating mean and standard deviation: {}% {}"
        print(str.format(0, '-'), end='\r')
        rot = 0
        for data_dir in sorted(dirs):
            data_dir_path = os.path.join(self.root_dir, data_dir)
            if os.path.isdir(data_dir_path):
                for image_file in glob.glob(os.path.join(data_dir_path,
                                                         '*.png')):
                    image = Image.open(image_file)
                    frame = np.array(image)
                    frame = frame / 255
                    for row in range(frame.shape[0]):
                        for col in range(frame.shape[1]):
                            count += 1
                            prev_mean = m
                            prev_s = s
                            val = frame[row, col]
                            t = val - prev_mean
                            m = prev_mean + t / count
                            s = prev_s + t * (val - m)
                    if rot < 10:
                        rot_sgn = 'Oooo'
                        rot += 1
                    elif rot < 20:
                        rot_sgn = 'oOoo'
                        rot += 1
                    elif rot < 30:
                        rot_sgn = 'ooOo'
                        rot += 1
                    elif rot < 40:
                        rot_sgn = 'oooO'
                        rot += 1
                    else:
                        rot = 0
                    print(str.format(round(i_num / num * 100),
                                     rot_sgn), end='\r')
            i_num += 1
        self.mean = torch.tensor(m)
        if count > 0:
            s = sqrt(s / count)
        self.std = torch.tensor(s)
        print(str.format(100, 'finished!'))

    def __save_mean_and_std(self):
        file_name = os.path.join(self.root_dir, 'mean_and_std.json')
        data = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist()
        }
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file)

    def __len__(self):
        # 2 - horizontally invert or not
        # 8 = 136 - 128 + 1 number of possible horizontal shifts
        # 8 = 136 - 128 + 1 number of possible vertical shifts
        if self.randomize_shake:
            n = self.randomize_shake_shift_all
            c = (n - 1) * 2 + 1
            count = 2 * self.shaken_frames_count * c * c
        else:
            count = self.extended_patch_size - self.patch_size + 1
            count = 2 * count * count

        if self.generate_sequences:
            return (len(self.sequences) * count)
        else:
            return (self.actual_sets_count * count)

    def __getitem__(self, idx):
        if self.generate_sequences:
            sequence = []
            for item_idx in self.__get_item_ids_for_sequence(idx):
                sequence.append(self.__getitem_helper(item_idx))
            return sequence
        else:
            return self.__getitem_helper(idx)

    def __getitem_helper(self, idx):
        # first select proper directory
        d1 = 2
        d2 = self.extended_patch_size - self.patch_size + 1
        if self.randomize_shake:
            n = (self.randomize_shake_shift_all - 1) * 2 + 1
            idx = idx // self.shaken_frames_count
            a2 = n * n
            dr2 = n
        else:
            a2 = d2 * d2
            dr2 = d2
        a1 = d1 * a2

        set_idx = idx // a1
        counter_id = 0
        range_id = 0
        found = False
        sets_counter = 0
        if set_idx >= self.actual_sets_count:
            raise IndexError("Index out of bound in __getitem__.")
        if self.use_all:
            set_type = "all"
        elif self.use_only_with_touch:
            set_type = "with_touch"
        else:
            set_type = "without_touch"
        while counter_id < len(self.counters) and not found:
            range_list = self.counters[counter_id][set_type]
            range_id = 0
            while range_id < len(range_list) and not found:
                r = range_list[range_id]
                sets_counter += r[1] - r[0] + 1
                if sets_counter > set_idx:
                    found = True
                else:
                    range_id += 1
            if not found:
                counter_id += 1
        if not found:
            raise IndexError("Index out of bound. Probable logic error in "
                             "__getitem__.")
        first, last = self.counters[counter_id][set_type][range_id]
        set_len = last - first + 1
        start_frame = first + set_idx - (sets_counter - set_len)
        rest = idx - (sets_counter - set_len - first + start_frame) * a1
        is_inverted = ((rest // a2) == 1)
        rest = rest % a2

        if self.randomize_shake:
            gdx = (rest // dr2) + (d2 - n) // 2
            gdy = (rest % dr2) + (d2 - n) // 2
        else:
            gdx = rest // dr2
            gdy = rest % dr2
        if set_type == "without_touch":
            frame_num = -1
        elif (
            set_type == "with_touch" or
            (first, last) in self.counters[counter_id]["with_touch"]
        ):
            touches = self.counters[counter_id]["touches"]
            i = 0
            frame_num = -1
            while i < len(touches) and frame_num == -1:
                touch = touches[i]
                if (
                    touch >= start_frame and
                    touch <= start_frame + self.set_len
                ):
                    frame_num = touch - start_frame
                i += 1
        else:
            frame_num = -1
        # print(idx, start_frame, is_inverted, dx, dy)

        dir_found = False
        data_idx = 0
        # print(sorted(os.listdir(self.root_dir)))
        for data_dir in sorted(os.listdir(self.root_dir)):
            data_dir_path = os.path.join(self.root_dir, data_dir)
            data_file = os.path.join(data_dir_path, 'info.xml')
            if os.path.isdir(data_dir_path) and os.path.isfile(data_file):
                if(data_idx == counter_id):
                    dir_found = True
                    break
                else:
                    data_idx += 1
        # print('DIR FOUND2:', data_dir_path)

        if not dir_found:
            raise RuntimeError(
                'There are less directories with data than the calculated '
                'directory index. There is an error in index calculation '
                'logic or directories were removed during program execution.')
        else:
            frames = []

            if self.with_patch_positions:
                positions = []
                patch_positions = {}
                try:
                    parsed_xml = xml.parse(data_file)
                    node = parsed_xml.find('positions')
                    patch_position_nodes = node.findall('pos')
                    for patch_pos in patch_position_nodes:
                        data = patch_pos.attrib
                        index = int(data['patch_id'])
                        patch_positions[index] = {'x': int(data['x']),
                                                  'y': int(data['y'])}
                except xml.ParseError as err:
                    print('ERROR: malformed XML file: ' +
                          data_file + '. Technical info: ' +
                          str(err))
                    exit()

            for i in range(start_frame, start_frame + self.set_len):
                image_name = os.path.join(data_dir_path, f'{i:02}' + '.png')
                # print('Opening:', image_name)
                image = Image.open(image_name)
                if self.randomize_shake:
                    # dx = random.randint(0, d2 - 1)
                    # dy = random.randint(0, d2 - 1)
                    s = self.randomize_shake_shift_all - 1
                    shift_x = random.randint(-s, s)
                    shift_y = random.randint(-s, s)
                    dx = gdx + shift_x
                    dy = gdy + shift_y
                else:
                    dx = gdx
                    dy = gdy
                image = image.crop((dx, dy, dx + self.patch_size,
                                    dy + self.patch_size))
                if self.with_patch_positions:
                    if i in patch_positions:
                        pos = patch_positions[i]
                        pos_x = pos['x'] + dx
                        if is_inverted:
                            pos_y = self.original_image_height - pos['y'] - dy
                        else:
                            pos_y = pos['y'] + dy
                        positions.append([pos_x, pos_y])
                    else:
                        print('ERROR: no position in xml for image with '
                              'number:', i)
                        exit()
                if is_inverted:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                frame = np.array(image)
                frame = frame / 255
                frames.append(frame)
            frames = np.asarray(frames)
            frames = frames[None, :, :, :]
            sample = torch.from_numpy(frames).type(torch.FloatTensor)
            sample.sub_(self.mean).div_(self.std)

            if self.ground_truth_as_index:
                frame_num -= self.ground_truth_shift
                if frame_num < 0:
                    frame_num = -1
                if self.with_patch_positions:
                    if self.normalize_patch_positions:
                        positions = self.__normalize_positions(positions)
                    return [sample, torch.FloatTensor(positions), frame_num]
                else:
                    return [sample, frame_num]
            else:
                ground_truth = torch.zeros(self.set_len)
                touch_index = (frame_num - self.ground_truth_shift)
                if (
                    touch_index >= 0 and
                    touch_index < self.set_len
                ):
                    ground_truth[touch_index] = 1
                if self.with_patch_positions:
                    if self.normalize_patch_positions:
                        positions = self.__normalize_positions(positions)
                    return [sample, torch.FloatTensor(positions), ground_truth]
                else:
                    return [sample, ground_truth]

    def __get_item_ids_for_sequence(self, idx):
        # d1 = 2
        # d2 = self.extended_patch_size - self.patch_size
        # if self.randomize_shake:
        #     a2 = self.shaken_frames_count
        # else:
        #     a2 = d2 * d2
        # a1 = d1 * a2

        # IMPORTANT: this method needs to be verified and corrected to be used
        # randomize_shake and randomize_shake_shift_all options were added
        # and must be properly incorporated into this method. Right now the
        # code is not used and is probably incorrect

        d1 = 2
        d2 = self.extended_patch_size - self.patch_size + 1
        if self.randomize_shake:
            n = (self.randomize_shake_shift_all - 1) * 2 + 1
            idx = idx // self.shaken_frames_count
            a2 = n * n
            dr2 = n
        else:
            a2 = d2 * d2
            dr2 = d2
        a1 = d1 * a2

        sequence_idx = idx // a1
        rest = idx % a1
        is_inverted = ((rest // a2) == 1)
        if self.randomize_shake:
            dx = random.randint(0, d2 - 1)
            dy = random.randint(0, d2 - 1)
        else:
            rest = rest % a2
            dx = rest // d2
            dy = rest % d2

        sequence = self.sequences[sequence_idx]
        counter_id = sequence['counter_id']
        first_frame_id = sequence['first_frame']
        last_frame_id = sequence['last_frame']
        item_ids = []
        offset = a2 * (1 if is_inverted else 0) + d2 * dx + dy
        if self.use_all:
            offset += a1 * self.counters[counter_id]['all_prev_count']
        elif self.use_only_with_touch:
            offset += a1 * self.counters[counter_id]['with_touch_prev_count']
        else:
            offset += a1 * self.counters[counter_id]['without_touch_prev_count']

        s_id = sequence_idx - 1
        while s_id >= 0 and self.sequences[s_id]['counter_id'] == counter_id:
            first_f = self.sequences[s_id]['first_frame']
            last_f = self.sequences[s_id]['last_frame']
            offset += a1 * (last_f - first_f + 1)
            s_id -= 1
        for frame_id in range(first_frame_id, last_frame_id + 1):
            f_idx = frame_id - first_frame_id
            item_id = a1 * f_idx + offset
            item_ids.append(item_id)
        return item_ids

    def __normalize_positions(self, positions):
        h = self.original_image_height - self.patch_size
        w = self.original_image_width - self.patch_size
        positions = list(map(
            lambda p: [p[0] / h, p[1] / w], positions))
        return positions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SqueezeNet predicting tool')
    parser.add_argument('-t', '--test_dir',
                        metavar='<directory with test data>',
                        required=True,
                        help='path to directory with test data')
    args = parser.parse_args()
    dataset = ShuttleClipsInSequencesDataset(args.test_dir,
                                             use_only_with_touch=True,
                                             generate_sequences=True)
    print('Datasets with generated sequences:')
    print('Data count only with touch ground:', len(dataset))
    dataset = ShuttleClipsInSequencesDataset(args.test_dir,
                                             use_only_without_touch=True,
                                             generate_sequences=True)
    print('Data count only without touch ground:', len(dataset))
    dataset = ShuttleClipsInSequencesDataset(args.test_dir,
                                             generate_sequences=True)
    print('Data count with all:', len(dataset))
    print('Sequences with ground touches and without touches are merged. '
          'Usually it reduces the number of sequences which get much longer. '
          'The count of all sequences is usually less than the count of '
          'sequences without touches or the count of sequences with touches.')
    dataset = ShuttleClipsInSequencesDataset(args.test_dir,
                                             use_only_with_touch=True,
                                             generate_sequences=False)
    print('Datasets without generated sequences:')
    print('Data count only with touch ground:', len(dataset))
    dataset = ShuttleClipsInSequencesDataset(args.test_dir,
                                             use_only_without_touch=True,
                                             generate_sequences=False)
    print('Data count only without touch ground:', len(dataset))
    dataset = ShuttleClipsInSequencesDataset(args.test_dir,
                                             generate_sequences=False)
    print('Data count with all:', len(dataset))

    # test every image
    # dataset = ShuttleClipsInSequencesDataset(args.test_dir,
    #                                          generate_sequences=True,
    #                                          randomize_shake=True)
    #
    # for sequence_id in range(len(dataset)):
    #     sequence = dataset[sequence_id]
    #     print(sequence_id)
    #     for frame_id, frame_set in enumerate(sequence):
    #         print(frame_id, end=' ')
    #     print()

    print('Datasets without generated sequences with randomized shake:')
    dataset = ShuttleClipsInSequencesDataset(args.test_dir,
                                             generate_sequences=False,
                                             randomize_shake=True,
                                             shaken_frames_count=10,
                                             randomize_shake_shift_all=3,
                                             frames_count=8)
    print('Data count with all:', len(dataset))

    dataset = ShuttleClipsInSequencesDataset(args.test_dir,
                                             generate_sequences=False,
                                             randomize_shake=True,
                                             shaken_frames_count=10,
                                             randomize_shake_shift_all=3,
                                             frames_count=8)

    # s = dataset[1234]
    # print(1234, s[1])
    for set_id in range(len(dataset)):
        s = dataset[set_id]
        print(set_id, s[1])

    # dataset = ShuttleClipsInSequencesDataset(
    #     args.test_dir,
    #     randomize_shake=False,
    #     extra_previous_frames_count=5,
    #     ground_truth_as_index=True,
    #     ground_truth_shift=0,
    #     frames_count=10
    # )
    # for set_id in range(len(dataset)):
    #     s = dataset[set_id]
    #     print(set_id, s[1])
