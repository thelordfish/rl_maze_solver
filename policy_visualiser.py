# policy_visualiser.py
"""
sorry for the insane amount of comments
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from mdp_env import ACTIONS

class PolicyVisualiser:

    BAR_MODE = "bars"
    ARROW_MODE = "arrows"

    def __init__(self, width: int, height: int) -> None:

        #1. State space dimensions (for layout)
        #These are the height and width of the maze in cells, also the exhaustive list of available states (as states are just (x,y) coordinates for now)
        self.width = width
        self.height = height
        
        #2. Visualisation mode (arrowmap or bar charts)
        self.mode = self.BAR_MODE #or ARROW_MODE

        #3. Set up window
        plt.ion()  # interactive mode on
        self.fig = plt.figure(figsize=(12, 8))

        #keep policy daya between switching layouts
        self.last_policy_data = None

        #building the intiial view:
        self.switch_layout()
        self.setup_events()


    def setup_events(self) -> None:
        # 'connect' a function to the figure's keyboard listener
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def on_key_press(self, event) -> None:
        # 'event.key' tells us exactly what button was pushed
        if event.key == "b":
            self.mode = self.BAR_MODE
            self.switch_layout()
        elif event.key == "a":
            self.mode = self.ARROW_MODE
            self.switch_layout()

    def update(self, policy_dict: Dict) -> None:
        self.last_policy_data = policy_dict

        if self.mode == self.BAR_MODE:
            self.update_bars(policy_dict)
        elif self.mode == self.ARROW_MODE:
            self.update_arrows(policy_dict)

    def switch_layout(self) -> None:
        self.fig.clear()  # clear existing view

        
        if self.mode == self.BAR_MODE:
            #64 bar charts, one per state, arranged in a grid matching the maze layout
            self.setup_bar_grid()
       

        if self.mode == self.ARROW_MODE:
            #the bar charts but argmaxed into a dominant arrow for each
            self.setup_arrowmap_grid()
         

    def setup_bar_grid(self) -> None:
        
        self.bar_grid = self.fig.subplots(self.height, self.width, sharex = True, sharey=True, )       #as initialised earlier self.fig = plt.figure(figsize=(12, 8))
                                                                                #sharey/x=True shares x and y scale
        self.bar_look_up = {}
        bar_colours = ['#3b82f6', '#22c55e', '#ef4444', '#f97316']
        action_labels = ['N', 'E', 'S', 'W']

        #plt.figure is a list of lists/ 2d numpy array, height*width, subplot boxes
        for inverse_row_idx, actual_row in enumerate(self.bar_grid):
            row_idx = (self.height-1) - inverse_row_idx        
            #taking the row index off the height (matplotlib starts y (idx 0) at the TOP left ( as a list of lists) not the bottom left)
                #-1 as it is 0 indexed so 8 would be out of bounds

            y = row_idx 
            for box_index, actual_box in enumerate(actual_row):
                x = box_index

                box_with_bar_chart = actual_box.bar(action_labels, [0, 0, 0, 0], color=bar_colours, width=0.7) #installing bar chart in box
                
                self.bar_look_up[(x, y)] = box_with_bar_chart   #can easily find and access particular bar charts with box coordinates

                self.format_bar_subplot(actual_box, x, y, inverse_row_idx, box_index, action_labels)

                

    def format_bar_subplot(self, actual_box, x, y, inverse_row_idx, box_index, action_labels):
        actual_box.set_ylim(0.0, 1.05)
        actual_box.set_yticks([])
        actual_box.set_xticks([])

        if inverse_row_idx == self.height - 1:
            actual_box.set_xticks(range(4))
            actual_box.set_xticklabels(action_labels, fontsize=5)

        if box_index == 0:
            actual_box.set_yticks([0, 0.5, 1.0])
            actual_box.set_yticklabels(['0', '.5', '1'], fontsize=5)

        actual_box.text(0.05, 0.92, f'{x},{y}',
                transform=actual_box.transAxes, fontsize=4,
                color='#999999', va='top', ha='left')
        
    def setup_arrowmap_grid(self) -> None:
        # create a single subplot rather than 64 so arrows arent awkwardly in boxes
        self.arrow_grid = self.fig.subplots(1, 1)

        x_range = range(0, self.width)
        y_range = range(0, self.height)

        #create coordinates for each cell: numpy.meshgrid creates two matrices X and Y. this is a much higher performance way to update things, and store indices than nested lists
        #allows for using set_UVC function later
        self.X, self.Y = np.meshgrid(x_range,y_range, indexing = 'xy')

        # create the arrows (pointing nowhere for now)
        self.quiver = self.arrow_grid.quiver(
            self.X, self.Y, #where arrows begin (centres of each cell)
            np.zeros_like(self.X), #intialize all the values as 0 making hieght*width matrices of 0s
            np.zeros_like(self.Y), 
            pivot='mid',#rotate at the centre of each#
            scale =25, # stops matplotlib from scaling arrows to fit the entire plot 
            width=0.007,
            cmap = 'viridis_r') #arrow colour scheme yellow to purpple. viridis_r is reversed so yellow is weak and purple strong

        # formatting 
        
        self.format_arrowmap()

    def format_arrowmap(self):
        self.arrow_grid.set_aspect('equal')
        self.arrow_grid.set_xlim(-0.5, self.width - 0.5)
        self.arrow_grid.set_ylim(-0.5, self.height - 0.5)
        self.arrow_grid.set_xticks([])
        self.arrow_grid.set_yticks([])

        for spine in self.arrow_grid.spines.values():
            spine.set_visible(False)

        for x in range(self.width + 1):
            self.arrow_grid.axvline(x - 0.5, color='#dddddd', linewidth=0.5)
        for y in range(self.height + 1):
            self.arrow_grid.axhline(y - 0.5, color='#dddddd', linewidth=0.5)

    def get_4_grids_from_nested_dict(self, policy_dict: Dict) -> Dict[str, np.ndarray]:
        """
        translates the policy dictionary into 4 separate grids (North, East, South, West)
        each grid contains the probability of that move
        """
        # we initialize a blank grid for compass directions
        direction_grids = {
            "N": np.zeros((self.height, self.width)),
            "E":  np.zeros((self.height, self.width)),
            "S": np.zeros((self.height, self.width)),
            "W":  np.zeros((self.height, self.width))
        }

        for (x, y), action_probs in policy_dict.items():
            for direction in direction_grids.keys():
                # fill the grid at (y, x) with the probability of that direction
                direction_grids[direction][y, x] = action_probs.get(direction, 0.0)

        return direction_grids

    def update_arrows(self, policy_data) -> None:
        grids = self.get_4_grids_from_nested_dict(policy_data)

        # creates a (4, Height, Width) stack
        stacked_probs = np.stack([grids["N"], grids["E"], grids["S"], grids["W"]])
        
    

        # np stack has three axes [Sheet, Row, Column], we want to argmax between sheets (directions), so axis = 0
        argmaxed_indices = np.argmax(stacked_probs, axis=0)

        #get confidence for arrow colour, max gives the value on the sheet (the probability), rather than the sheet number like argnax
        confidence = np.max(stacked_probs, axis=0) 

        horizontal_push = np.array([0, 1, 0, -1]) # ([N, E, S, W]) so if east is 1, its +1, west is -1, 
        vertical_push   = np.array([1, 0, -1, 0])# same here

        # use the argmaxed_indices to pick the right push for every cell
        # no loops, just mapping
        u_directions = horizontal_push[argmaxed_indices]
        v_directions = vertical_push[argmaxed_indices]

        # update the existing visual object
        self.quiver.set_UVC(u_directions, v_directions, confidence)
        self.fig.canvas.draw_idle()


    def update_bars(self, policy_dict: Dict) -> None:
        """
        Updates the 64 tiny bar charts.
        policy_dict: {(x, y): {'North': p1, 'East': p2, ...}}
        """
        self.last_policy_data = policy_dict #store policy values between switches

        for (x, y), action_probs in policy_dict.items():
            if (x, y) in self.bar_look_up:
                # We stored the bar container in our lookup table
                bar_container = self.bar_look_up[(x, y)]
                
                # Get probabilities in the order we defined in setup: N, E, S, W
                # These match the [0, 1, 2, 3] x-axis of the bar charts
                new_heights = [
                    action_probs.get("N", 0),
                    action_probs.get("E", 0),
                    action_probs.get("S", 0),
                    action_probs.get("W", 0)
                ]
                
                # Update the heights of the rectangles in the bar chart
                for rect, h in zip(bar_container, new_heights):
                    rect.set_height(h)

        self.fig.canvas.draw_idle()