# Sudoku
Sudoku Application used to :
* Detect Sudoku board using CV techniques
* Classify digits using a model trained on 'printed' digits instead of 'handwritten'
* Solve the predicted board
* Display an UI over the images


## Table of contents
* [Preview](#preview)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Status](#status)


## Preview
![Preview](./preview.gif)

## Technologies
* Python - version 3.8.3
* OpenCV - version 4.2.0
* Tensorflow - version 2.3.0 (GPU)

## Setup
Inc.

## How to run
Main program
`python main.py`

## Features
* [BoardDetection & Tracker](./Utils/)
* [Digit Classifier trained on personnal data](./Model)
* [Solver](./SudokuDLX)
* [AugmentedReality](./AR)

TODO:
* Kivy Android app
* Enhance detection/classification
* Make Real AR (with Rt matrix enhanced over time instead of KeyPoints tracker)
* Make everything work on tflight for android version

## Status
Project is: _in progress_
