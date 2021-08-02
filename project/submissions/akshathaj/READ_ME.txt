README file:
------------

* The project folder contains the following: src\, examples\, tests\, data\, config.toml and READ_ME.txt
* All the main codes are in the folder 'tests'.
* All the 6 files in the src\, config.toml and corresponding data\* file are necessary for the scripts in the tests\ folder to run.
* The route screenshots for different arenas are in the folder 'examples'.
* The tests\ipynb_files contain ipynb files
* The tests\py_files contain py files
* There are 4 codes as follows
	1. tests\*\astar_grid_4_path codes the 4 way lattice and uses corners to make route 
		data: data\astar_grid.npy
		output: examples\astar_grid.png
	2. tests\*\arena_1_tracking codes the route for the following. No obstacle is added after path planning
		data: data\arena_1.map
		output: examples\arena_1.png
	3. tests\*\arena_2_tracking codes the route for the following. No obstacle is added after path planning
		data: data\arena_2.map
		output: examples\arena_2.png
	2. tests\*\arena_3_tracking_with_obstacle codes the route for the following. An obstacle is added after path planning for collision checking in DWA
		data: data\arena_3.map
		output: examples\arena_3_with_obstacle.png
* The astar path is in red colour and the dwa tracked path is in green colour. Note that only for the red path for examples\astar_grid is the trajectory_planning output. This is the case only for examples\astar_grid.
