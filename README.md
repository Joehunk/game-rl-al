# Installation
1. Install VS Code and open this folder.
2. Create a Python virtual environment using venv (tested with 3.11). Search for venv in the VS Code command palette.
3. Create a Python terminal (command palette). Important: use 'Python: create terminal' 
   not the normal terminal to ensure that the terminal opens in the virtual environment.
4. From the terminal: 
    * `pip install mss`
    * Go to [this web site](https://pytorch.org/get-started/locally/) and copy paste the pip command to install PyTorch Stable and CUDA 12.x into the terminal.

# Aligning
1. Launch SCUM in windowed mode
2. Choose a resolution and remember it. Use this every time.
3. Move the SCUM window such that it is as close to the top left of the monitor as possible.
4. You will need to edit `settings.py` with the bounding box of the lockpick UI.
    * You can use a program [like this](https://sourceforge.net/projects/mpos/) to find mouse coordinates to assist with this.
5. Once you have a good guess, run `python scum_game.py` from the command line
    * Ensure the lockpick UI is visible (not hidden under another window).
6. This will create a file `test_align.png`. Ensure this image looks similar to `test_align_example.png`. Change `settings.py` 
   and repeat step 5 until it does
    * Scale does not matter, but alignment does. Ensuire the borders are similar. It doesn't have to be pixel-perfect but try to get close. 

If you ever start getting bad results, it might be a good idea to re-align. Start with step 5 and see if the produced
`test_align.png` looks okay.
