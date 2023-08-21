from pretty_midi import PrettyMIDI
from typing import List, Dict, Union, Tuple
from loguru import logger as lg
from pathlib import Path
import numpy as np


class Playability:
    def __init__(
        self,
        MIDIFile: Union[PrettyMIDI, Union[str, Path]],
        fret_threshold: int = 4,
        hand_threshold: int = 5,
        error_distubution: Tuple[float, float, float, float] = (
            0.3,
            0.3,
            0.2,
            0.1,
            0.1,
        ),
        transpose: bool = True,
    ) -> None:
        """Simulating hand positions while playing a 6 stringed (at the moment), normal tuning (at the moment) guitar.
        @Params
        @MIDIFile: a PrettyMIDI object or a path of the .mid file to process.

        @threshold: maximum length of the notes apart in frets numbers. Although it depends on the position, a skilled player can
        play notes that are 5 frets apart but in order to be more calculate the score more generally, threshold is set to 4.

        @hand_threshold: max fret number that a hand can move between two different shapes between two time steps. Default to 5.

        @error_distribution: (a,b,c,d,e) where a is for fret_error, b is for hand_error, c is for max_min_pitch error,
        d is for max 6 playable strings error and e is for impossible to play (some edgy cases where it looks like it's possible to play
        but in fact it's not.) error.

        @transpose: whether to transpose the pitches that doesn't fall into the possible pitch range. If set to True, will transpose
        the notes into the nearest octave. If set to false, will remove those notes. In both cases the error rate will be calculated.

        In this implementation time_step indicates the begining of each note that have different start times.
        """
        # Todo: write a function that can produce the above result. With such function, we'll be able to change tuning as well.
        self.guitar_fretboard = {
            40: [0, -1, -1, -1, -1, -1],  # E2
            41: [1, -1, -1, -1, -1, -1],
            42: [2, -1, -1, -1, -1, -1],
            43: [3, -1, -1, -1, -1, -1],
            44: [4, -1, -1, -1, -1, -1],
            45: [5, 0, -1, -1, -1, -1],
            46: [6, 1, -1, -1, -1, -1],
            47: [7, 2, -1, -1, -1, -1],  # B2
            48: [8, 3, -1, -1, -1, -1],
            49: [9, 4, -1, -1, -1, -1],
            50: [10, 5, 0, -1, -1, -1],
            51: [11, 6, 1, -1, -1, -1],
            52: [12, 7, 2, -1, -1, -1],
            53: [13, 8, 3, -1, -1, -1],
            54: [14, 9, 4, -1, -1, -1],
            55: [15, 10, 5, 0, -1, -1],
            56: [16, 11, 6, 1, -1, -1],
            57: [17, 12, 7, 2, -1, -1],
            58: [18, 13, 8, 3, -1, -1],
            59: [19, 14, 9, 4, 0, -1],
            60: [20, 15, 10, 5, 1, -1],
            61: [21, 16, 11, 6, 2, -1],  # C#4
            62: [-1, 17, 12, 7, 3, -1],
            63: [-1, 18, 13, 8, 4, -1],
            64: [-1, 19, 14, 9, 5, 0],
            65: [-1, 20, 15, 10, 6, 1],
            66: [-1, 21, 16, 11, 7, 2],
            67: [-1, -1, 17, 12, 8, 3],
            68: [-1, -1, 18, 13, 9, 4],
            69: [-1, -1, 19, 14, 10, 5],
            70: [-1, -1, 20, 15, 11, 6],
            71: [-1, -1, 21, 16, 12, 7],
            72: [-1, -1, -1, 17, 13, 8],
            73: [-1, -1, -1, 18, 14, 9],
            74: [-1, -1, -1, 19, 15, 10],
            75: [-1, -1, -1, 20, 16, 11],
            76: [-1, -1, -1, 21, 17, 12],
            77: [-1, -1, -1, -1, 18, 13],
            78: [-1, -1, -1, -1, 19, 14],
            79: [-1, -1, -1, -1, 20, 15],
            80: [-1, -1, -1, -1, 21, 16],
            82: [-1, -1, -1, -1, -1, 17],
            81: [-1, -1, -1, -1, -1, 18],
            83: [-1, -1, -1, -1, -1, 19],
            84: [-1, -1, -1, -1, -1, 20],
            85: [-1, -1, -1, -1, -1, 21],  # 21th fret
        }
        self.MAX_PITCH = max(list(self.guitar_fretboard.keys()))
        self.MIN_PITCH = min(list(self.guitar_fretboard.keys()))

        if fret_threshold <= 0:
            lg.critical("Fret threshold can't be 0 or negative")
            raise Exception
        if hand_threshold <= 0:
            lg.critical("Hand threshold can't be 0 or negative")
            raise Exception

        self.midi_file = (
            PrettyMIDI(MIDIFile) if isinstance(MIDIFile, (str, Path)) else MIDIFile
        )
        self.fret_threshold = fret_threshold
        self.hand_threshold = hand_threshold

        self.hand_threshold_error = 0
        self.fret_threshold_error = 0
        self.impossible_to_play = 0

        # times steps to look at each iteration.
        self.times_pitches = self.create_vocab(self.midi_file)
        # {() : [pitch_0, pitch_1, max to six]}
        # lg.info(f"times_pitches: {self.times_pitches}")

        # Process the midi file by removing the pitches that aren't inside the pitch interval
        # and removing some the notes that are more than 6 notes present in that time interval.
        if transpose == False:
            (
                self.cleaned_times_pitches,
                self.pitch_error,
                self.number_of_D2,
            ) = self.remove_nonguitar_pitch(self.times_pitches)
        else:
            (
                self.cleaned_times_pitches,
                self.pitch_error,
                self.number_of_D2,
            ) = self.transpose_piece(self.times_pitches)

        # lg.info(f"cleaned_times_pitches: {self.cleaned_times_pitches}")
        self.cleaned_times_pitches, self.six_error = self.return_six_error(
            self.cleaned_times_pitches
        )

        # hand position for each time step. It's calculated as the average of the farthest notes
        self.hand_position = [-1.0]

        # [[[1,2],[2,5],[4,2]], [[1,2],[2,4],[3,2]]] wholesong[timesteps[pitches]].
        # self.note_locations_A = []

        # Another approach could be like this: [[2, -1, -1, 3, 2, 0], [0, -1, -1, 2, 4, 2], ...] wholesong[timesteps[_,_,_,_,_,_]]
        self.time_note_locations = self.iterate_notes(self.cleaned_times_pitches)
        # lg.info(f"time_note_locations: {self.time_note_locations}")
        # itprint(self.time_note_locations)
        # itprint(self.time_note_locations[824])

        # self.playability = ((1-(((error_distubution[0] * self.fret_threshold_error) + (error_distubution[1] * self.hand_threshold_error) + (error_distubution[2] * self.))/len(self.cleaned_times_pitches)))) * 100
        self.playability_rate = (
            100
            - (
                self.pitch_error * error_distubution[2]
                + self.six_error * error_distubution[3]
            )
            - (
                100
                * error_distubution[0]
                * (self.fret_threshold_error / len(self.time_note_locations))
            )
            - (
                100
                * error_distubution[1]
                * (self.hand_threshold_error / len(self.time_note_locations))
            )
            - (
                100
                * error_distubution[4]
                * (self.impossible_to_play / len(self.time_note_locations))
            )
        )

        # self.show_stats(True, True)

    def create_vocab(self, mid: PrettyMIDI) -> Dict[Tuple[float, float], List[int]]:
        """We wish to have a structure like this: [[4,5,9], [15], [2], [14], ...].
        Here each list indicates the created time_interval and inner integers are the pitches
        presence in that time_interval. We iterate the algorithm at each time step."""

        # we should be creating the intervals like this: [[start, min]]
        note_start_ends_randomized = []
        for note in mid.instruments[0].notes:
            if note.start not in note_start_ends_randomized:
                note_start_ends_randomized.append(note.start)
            if note.end not in note_start_ends_randomized:
                note_start_ends_randomized.append(note.end)

        min_intervals = {}  # every little interval we wish to look for.
        # Not (start, start) or (end, start) or something like that. We are looking at everything.

        note_times = sorted(
            note_start_ends_randomized
        )  # sort the times (both starts and ends)
        for i in range(len(note_start_ends_randomized) - 1):
            min_intervals[
                (note_times[i], note_times[i + 1])
            ] = 0  # initializing the dictionary

        for time1, time2 in min_intervals.keys():
            list_to_append = []
            # lg.info(f"time1: {time1}, time2: {time2}")
            for note in mid.instruments[0].notes:
                if time1 >= note.start and time2 <= note.end:
                    list_to_append.append(note.pitch)
                else:
                    pass

                min_intervals[(time1, time2)] = list_to_append

        # E.g. :{(start, min_0): [65,71], (min_1, min_2): [34, 7, 8, 9]]}
        return min_intervals

    def create_time_steps(self, obj: PrettyMIDI):
        """Creates the time steps using pretty_midi.Notes.start.
        DEPRECIATED.
        """
        raise NotImplementedError

    def iterate_notes(self, times_pitches_param: Dict[Tuple[float, float], List[int]]):
        """'Big brain algorithm...'
        1. Create all the possible permutations of guitar positions with the given pitch values.
        e.g. [[1,5], [2,7], [4,8]...(max 6)]
        2. Then remove the ones that are using the same string.
        3. Calculate the maximum length between notes on the fretboard for each iteration.
        4. Take the minimum of them.
        5. Update the note_locations and hand_position.
        6. Return max distance for playability_score() to use.
        """

        time_bestlocs_dict = times_pitches_param.copy()
        for i in range(len(times_pitches_param.values())):
            notes = times_pitches_param[list(times_pitches_param.keys())[i]]
            if notes == []:
                # pos=[] because is_rest=True.
                self.update_hand_pos(pos=[], is_rest=True)
                continue
            else:  # if notes exists
                container_list: List[List[List[int]]] = []
                for pitch in notes:
                    # lg.info(f"{pitch} : {self.create_note_on_string(self.guitar_fretboard[pitch])}")
                    container_list.append(
                        self.create_note_on_string(self.guitar_fretboard[pitch])
                    )
                merged_lists = self.merge_lists(container_list)[0]

                if merged_lists == []:
                    # pitch_value: [number_of_occurence_in_notes, number_of_possible_places_that_we_can_play]

                    notes, temp_dict = self._edge_handler1(notes)

                    merged_lists = self._create_possible_places(notes)

                    # If merged_lists still empty after these, just remove the octaves by using np.unique
                    if merged_lists == []:
                        notes = np.unique(notes)
                        merged_lists = self._create_possible_places(notes)

                time_bestlocs_dict[list(time_bestlocs_dict.keys())[i]] = self.L2(
                    merged_lists
                )

        return time_bestlocs_dict

    def _edge_handler1(
        self, notes: List[int]
    ) -> Tuple[List[int], Dict[int, Tuple[int, int]]]:
        """Here is a nice edge case: If there are 3 (or more) same notes for some reason (probably because of the midi encoding, it duplicates stuff?)
        you may not play on the guitar (because not all the notes have 3 or more places to play. This could go for 2 notes as well.). Here's
        a debug print for you to understand the issue better:

        2023-08-13 21:19:56.440 | DEBUG    | __main__:iterate_notes:245 - [47, 47, 47]
        container_list: [[[7, -1, -1, -1, -1, -1], [-1, 2, -1, -1, -1, -1]], [[7, -1, -1, -1, -1, -1], [-1, 2, -1, -1, -1, -1]], [[7, -1, -1, -1, -1, -1], [-1, 2, -1, -1, -1, -1]]]
        self.merge_lists(container_list)[0]: []

        3 notes, 2 possibilities; all iterations fails at _check_occurence()."""
        # To catch the mention error.
        # If this happens, we are going to discard minimum amount of octaves from the notes
        # and return the possible ones.

        # lg.debug("Handling an edge case...")
        temp_dict: Dict[int, Tuple[int, int]] = {}

        for note in notes:
            if note in list(temp_dict.keys()):
                temp_dict[note] = [
                    temp_dict[note][0] + 1,
                    temp_dict[note][1],
                ]
            else:
                temp_dict[note] = [
                    1,
                    len(self.retr_indexes_elems(self.guitar_fretboard[note])[0]),
                ]
        for note in temp_dict:
            if temp_dict[note][0] >= temp_dict[note][1]:
                # lg.debug(f"All notes are: {notes}")
                # lg.debug(f"Problematic note is: {note}")
                # lg.debug(
                #    f"Possible places to play that note are: {self.guitar_fretboard[note]}"
                # )
                for i in range(temp_dict[note][0] - temp_dict[note][1]):
                    notes.remove(note)

        new_notes = []
        for pitch in temp_dict:
            for i in range(temp_dict[pitch][0]):
                new_notes.append(pitch)
        # lg.debug(f"Old notes: {notes}")
        # lg.debug(f"New notes: {new_notes}")

        self._update_threshold_errors(impossible=True)

        return new_notes, temp_dict

    def _create_possible_places(self, notes: List[int]):
        container_list: List[List[List[int]]] = []

        for pitch in notes:
            container_list.append(
                self.create_note_on_string(self.guitar_fretboard[pitch])
            )
        return self.merge_lists(container_list)[0]

    def create_note_on_string(self, pitch_repr: List[int]) -> List[List[int]]:
        """Processes the data on self.guitar_strings to create the notes on single string.
        E.g. [-1, -1, 21, 16, 12, 7] => [[-1,-1,-1,21,-1,-1],[-1,-1,-1,-1,16,-1], ...]
        :Params:
        :pitch_repr: pitch representation on the guitar just like stated above.

        Returns List[List[], List[], ...] as mentioned.
        """
        # n=0
        base_repr = [-1, -1, -1, -1, -1, -1]
        final_list = []

        # for j in pitch_repr:
        #    if j != -1:
        #        n+=1
        indices, elements = self.retr_indexes_elems(pitch_repr)

        for i in range(len(indices)):
            base_repr_copy = base_repr.copy()
            base_repr_copy.pop(indices[i])
            base_repr_copy.insert(indices[i], elements[i])
            final_list.append(base_repr_copy)

        return final_list

    def merge_lists(self, list_of_lists: List[List[List[int]]]):
        """With the help of merge_two_lists() method, this method merges multiple lists. list_of_lists param may include more than two lists
        which each inner lists indicates the single notes' places and the outer list indicates this process for all notes.
        I'm checking the problem of being in the same string in this method rather than in the merge_two_lists().
        Also we need to look at the situation where there is only 1 note.
        """

        # list_of_lists shouldn't be 1. That case should be handled in iterate_notes()
        # lg.debug(list_of_lists)
        # assert len(list_of_lists) != 1

        list_of_lists2 = list_of_lists.copy()
        # lg.info(f"Length of the list_of_lists: {len(list_of_lists2)}")

        while len(list_of_lists2) >= 2:
            list1: List[List[int]] = list_of_lists2[0]
            list2: List[List[int]] = list_of_lists2[1]
            list_merged = []

            for elem1 in list1:
                ind1, elements1 = self.retr_indexes_elems(elem1)
                for elem2 in list2:
                    ind2, elements2 = self.retr_indexes_elems(elem2)
                    if self._check_occurence(ind1, ind2):
                        # if there are occurence between two index lists this means
                        # they are on the same string. So continue the loop without adding them the main merged_list.
                        continue
                    else:
                        list_merged.append(self.merge_two_lists([elem1, elem2]))

            # remove the first and second list to add their merged one.
            list_of_lists2 = list_of_lists2[2:]
            list_of_lists2.insert(0, list_merged)

        # np is totally cool with this edge ase though. klsdfjlksjdf
        return np.unique(list_of_lists2, axis=-2).tolist()

    def merge_two_lists(self, list_of_lists: List[List[int]]):
        """Merges two lists like [-1, -1, -1, -1, 12, -1, -1], [-1, -1, -1, -1, 7, -1] => [-1, -1, -1, -1, 12, 7, -1]
        We are not considering the intersection here.
        Note: I'm going to merge the two then create one then merge another one with the created one and so on. With this approach
        I'll iterate all the possibilities while obeying the rules of not playing two notes at the same time on a single string.
        """

        if len(list_of_lists) != 2:
            lg.critical(
                "Number of lists to merge should be 2. Please provide List[List[int], List[int]] as list_of_lists parameter."
            )
            raise Exception

        base_repr = [-1, -1, -1, -1, -1, -1]
        ind1, elem1 = self.retr_indexes_elems(list_of_lists[0])
        ind2, elem2 = self.retr_indexes_elems(list_of_lists[1])

        base_repr = self.iterate_insertion(base_repr, ind1, elem1)
        base_repr = self.iterate_insertion(base_repr, ind2, elem2)

        return base_repr

    def retr_indexes_elems(self, a_list: List[int]) -> Tuple[List[int], List[int]]:
        """Returns the index of the element which is not -1 in a list shaped like [-1, -1, -1, -1, 12, -1, -1]."""
        index = [-1]
        element = [-1]
        # lg.debug(a_list)
        for i, elem in enumerate(a_list):
            if elem != -1:
                if -1 in index and -1 in element:
                    # All these -1 stuff because I don't want to mess up with the Tuple[List[int], List[int]] with optionals
                    index.remove(-1)
                    element.remove(-1)
                index.append(i)
                element.append(elem)

        # I'm even putting this statement here to show something like this never going to happen
        assert index != [-1]

        return index, element

    def iterate_insertion(self, base: List[int], ind: List[int], elems: List[int]):
        """Iterate the insertion process over multiple indexes and elements."""

        assert len(ind) == len(elems)

        for i in range(len(ind)):
            base.pop(ind[i])
            base.insert(ind[i], elems[i])

        return base

    def _check_occurence(self, list1: List[int], list2: List[int]) -> bool:
        """Checks whether there are any occurence between two List[int]. If occurence exists returns True, otherwise False"""
        for i in list1:
            if i in list2:
                return True

        return False

    def update_hand_pos(self, pos: List[int], is_rest=False):
        """Updates the hand position. Averages the positions of the notes played on the guitar.
        If is_rest=True, appends the last element of the self.hand_position. This parameter can be used
        by open strings first approach as well (we don't change hand position while playing only open strings.)
        """
        if is_rest == True:  # This means we are at rest
            self.hand_position.append(
                self.hand_position[-1]
            )  # Our hand position stays the same at this time_step
        else:
            indices, elements = self.retr_indexes_elems(pos)
            self.hand_position.append((min(elements) + max(elements)) / 2)

    def _maxmin_nonzero(self, a_list: List[int]):
        """Returns the max and min elements which are nonzero of a list respectively."""

        a_list2 = a_list.copy()
        for i in a_list2:
            if i == 0:
                a_list2.remove(0)
            elif i == -1:
                a_list2.remove(-1)
            else:
                continue

        return max(a_list2), min(a_list2)

    def _all_zeros(self, a_list: List[int]):
        """Returns true if a list is full of 0's (and -1's). In other words, the position is an completely open string.
        Otherwise False."""

        for elem in a_list:
            if elem != -1 and elem != 0:
                return False

        return True

    def L2(self, pos: List[List[int]]) -> List[int]:
        """Finds the max length of the farthest two notes in the guitar fretboard.
        Decision: there could be a less distance between another part of the fretboard but if that's too far (> hand_threshold)
        from the current hand position we don't prefer it. More specifically, we increase the fret_threshold minimally (could be 1,2,3, so on.)
        and look at the possible shapes around the hand_position[-1]. If the problem persist though, we select the farther
        position. We don't increase the threshold by 1 again, it's only once.

        Edge Case 1: In the start of the song, our hand position could be anywhere.
        If there are more than one suitable position for both fret_threshold and hand_threshold, take the one that's closest to the start
        of the neck.

        Edge Case 2: Well it's not an edge case, I've just missed it somehow! We don't need to consider open strings because we can play
        them everywhere. So open strings shouldn't be considered while looking at min and max

        I'm taking the 'open string first' approach. If it's possible to play the open string, we'll just play that.
        This is practically bad becuase you might lose sustain but it can played like this and it's not an error.
        """
        # assert pos != [], "Some error that I can't define."

        if pos == []:
            # Todo: try to catch every edge case.
            # lg.error("Well, pos=[] again...")
            self._update_threshold_errors(impossible=True)
            self.update_hand_pos([], True)
            return []

        # No need to big brain it, just only one option
        if len(pos) == 1:
            self.update_hand_pos(pos[0])
            return pos[0]

        # if it's only on open strings just play it. 'open strings first' approach
        for position in pos:
            if self._all_zeros(position):
                # lg.debug(f"Open strings first: {position}")
                self.update_hand_pos(pos=[], is_rest=True)
                return position

        # pos is coming as [] somehow
        note_distances = []
        possible_hand_positions = []

        for position in pos:
            indices, elements = self.retr_indexes_elems(position)
            # finding the distance between the max fret and min fret. (other than open strings.)

            # if self._all_zeros(elements):

            note_distances.append(
                self._maxmin_nonzero(elements)[0] - self._maxmin_nonzero(elements)[1]
            )
            possible_hand_positions.append(
                (self._maxmin_nonzero(elements)[0] + self._maxmin_nonzero(elements)[1])
                / 2
            )

        if min(note_distances) > self.fret_threshold:
            self._update_threshold_errors(fret=True)

        min_dist_indices = []
        for index, elem in enumerate(note_distances):
            if elem == min(note_distances):
                min_dist_indices.append(index)

        if self.hand_position == [-1]:
            min_hand_index = possible_hand_positions.index(min(possible_hand_positions))
            self.update_hand_pos(pos[min_hand_index])
            return pos[min_hand_index]

        else:  # there is a previous hand position
            # iterate every possible minimum positions to find a suitable one for hand_threshold param
            for index in min_dist_indices:
                if (
                    abs(possible_hand_positions[index] - self.hand_position[-1])
                    < self.hand_threshold
                ):  # good case
                    self.update_hand_pos(pos[index])
                    return pos[index]

            self._update_threshold_errors(hand=True)

            # if new hand position will be too far from the last hand position, increase the fret_threshold minimally
            # getting the indices of not the minimals but minimals + min_value
            min2_dist_indices = []
            for i, distance in enumerate(sorted(note_distances)):
                if distance == min(note_distances):
                    pass
                else:
                    min2_dist_indices.append(i)

            for index in min2_dist_indices:
                if (
                    abs(possible_hand_positions[index] - self.hand_position[-1])
                    < self.hand_threshold
                ):  # kinda-good case
                    self.update_hand_pos(pos[index])
                    return pos[index]

            # if the problem persist we look at all the min fret positions and take the one which has the minimum distance
            # from the previous hand position

            min_hand_distances = []
            for index in min_dist_indices:
                min_hand_distances.append(
                    abs(self.hand_position[-1] - possible_hand_positions[index])
                )

            min_hand_index = min_hand_distances.index(min(min_hand_distances))
            self.update_hand_pos(pos[min_hand_index])
            return pos[min_hand_index]

    def _update_threshold_errors(self, fret=False, hand=False, impossible=False):
        if fret == True:
            self.fret_threshold_error += 1
        if hand == True:
            self.hand_threshold_error += 1
        if impossible == True:
            self.impossible_to_play += 1

    def return_six_error(
        self, times_pitches: Dict[Tuple[float, float], List[int]]
    ) -> Tuple[Dict[Tuple[float, float], List[int]], float]:
        """Return the error rate of the file if such error exists. It is the same as the playability
        function that I've already implemented.
        Returns the playability rate if there are errors otherwise 100."""
        times_pitches_copy = times_pitches.copy()
        time_error = 0

        for i in range(len(times_pitches_copy)):
            if (
                len(list(times_pitches_copy.values())[i]) > 6
            ):  # Guitar's maxiumum string number is set to 6. (Not creating a variable for that rn.)
                time_error += (
                    list(times_pitches_copy.keys())[i][1]
                    - list(times_pitches_copy.keys())[i][0]
                )
                # lg.debug("Removing some from soo many notes")
                times_pitches_copy[list(times_pitches_copy.keys())[i]] = sorted(
                    list(times_pitches_copy.values())[i], reverse=True
                )[:5]
                # Todo: definetelly change this. You amy remove the octaves (duplicate notes) maybe but not all the bass notes.
                # if there are more than 6, remove the bass part as much as needed.

        return times_pitches_copy, (time_error / len(times_pitches)) * 100

    def remove_nonguitar_pitch(
        self, times_pitches: Dict[Tuple[float, float], List[int]]
    ) -> Tuple[
        Dict[Tuple[float, float], List[int]], float, int
    ]:  # : Dict[Tuple[float, float]: List[int]]
        """If a midi file has pitches other than the normal tuned guitars pitches which is 40-85 (included),
        1) Calculate how many
        2) Remove them
        3) Return an error rate of it."""

        cleaned_times_pitches = times_pitches.copy()

        err = 0
        total_notes = 0
        note_D = 0
        for i, (time1, time2) in enumerate(cleaned_times_pitches):
            for pitch in list(cleaned_times_pitches.values())[i]:
                total_notes += 1
                notes_cleaned = list(cleaned_times_pitches.values())[i].copy()
                if pitch > self.MAX_PITCH or pitch < self.MIN_PITCH:
                    if pitch == 38:
                        note_D += 1
                    err += 0
                    notes_cleaned.remove(pitch)
                cleaned_times_pitches[(time1, time2)] = notes_cleaned

        return cleaned_times_pitches, (err / total_notes) * 100, note_D

    def _transpose_note(
        self, note_pitch: int, pitch_range: List[int] = [40, 85]
    ) -> Tuple[int, bool]:
        """Given a note pitch, increase or decrese the pitch by 12 until it falls between possible pitch_range
        :Params:
        :note_pitch: pitch value to work on.
        :pitch_range: pitch range to work on. [min_pitch, max_pitch]

        Returns:
        (transposed_pitch, True)
        (not_transposed_pitch, False) already between the interval.
        """

        if note_pitch < pitch_range[0]:  # smaller than
            if (pitch_range[0] - note_pitch) % 12 == 0:
                return (pitch_range[0], True)
            else:
                return (
                    note_pitch + 12 * (1 + ((pitch_range[0] - note_pitch) // 12)),
                    True,
                )

        elif note_pitch > pitch_range[1]:  # greater than
            if (note_pitch - pitch_range[1]) % 12 == 0:
                return (pitch_range[1], True)
            else:
                return (
                    note_pitch - 12 * (1 + ((note_pitch - pitch_range[1]) // 12)),
                    True,
                )

        else:  # it falls between the range
            return (note_pitch, False)

    def transpose_piece(
        self,
        times_pitches: Dict[Tuple[float, float], List[int]],
        pitch_range: List[int] = [40, 85],
    ) -> Tuple[Dict[Tuple[float, float], List[int]], float, int]:
        """Tranpose pitches that are greater than the max pitch and less than the min pitch to nearest possible octave.
        :Params:
        :input_paths: input paths. Use it with glob().
        :output_dir: output dir to save the midis.
        :pitch_range: [min_pitch, max_pitch]. Default to [40,85] which is for classical guitar.
        """
        manipulation_num = 0

        tranposed_times_pitches = times_pitches.copy()
        note_D = 0
        total_notes = 0

        for i, (time1, time2) in enumerate(tranposed_times_pitches):
            for pitch_pos, pitch in enumerate(
                list(tranposed_times_pitches.values())[i]
            ):
                if pitch == 38:
                    note_D += 1
                total_notes += 1
                new_pitch, is_tranposed = self._transpose_note(pitch, pitch_range)
                if is_tranposed:
                    manipulation_num += 1
                    tranposed_times_pitches[(time1, time2)][pitch_pos] = new_pitch

        return (tranposed_times_pitches, (manipulation_num / total_notes) * 100, note_D)

    def show_stats(self, hand: bool = False, time_step: bool = False) -> None:
        if time_step:
            lg.info("self.time_note_locations data is:")
        if hand:
            lg.info(f"Hand position data is: \n{self.hand_position}")
        lg.info(
            "Statistics"
            + f"\nPlayability rate: {self.playability_rate}, \n"
            + f"Fret Error: {self.fret_threshold_error}, \n"
            + f"Hand Position Error: {self.hand_threshold_error}, \n"
            + f"Pitch Interval Error: {self.pitch_error}, \n"
            + f"Max 6 String Error: {self.six_error}, \n"
            + f"Impossible to Play Error: {self.impossible_to_play}"
        )
