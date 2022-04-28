# CARLA-behavioural-planner

## How to run
To run Carla Server whit this script, you need the following settings:

    .\CarlaUE4.exe /Game/Maps/Town01 -carla-server -windowed -benchmark -fps=30

To run the script consider the following arguments:

    python main.py -ss START1,STOP1.START2,STOP2.[...] -nv 10,100,[...] -np 10,100.[...] -sv 1,2,67,[...] -sp 1,4,77,[...]

where
 * ```ss``` is an argument who require the stary waypoint and the stop waypoint, separated by a comma. 
 If you would try more than one path you can indicate more than one path, separating them by a dot. For example,
 ```python main.py -ss 88,28.26,14.6,8``` will try the scene in the 3 cases.
 * ```nv``` is the number of pedestrians to generate. Multiple chooses must be separated by a comma like "1,3"
 * ```np``` is the number of vehicles to generate. Multiple chooses must be separated by a comma like "1,3".
 * ```sv``` is the seed to generate vehicles. Multiple chooses must be separated by a comma like "1,3"
 * ```sp``` is the seed to generate pedestrians. Multiple chooses must be separated by a comma like "1,3"

All the possible combination will be executed. A report of the execution can be found in the log folder.

In absence of arguments, the default parameters are declared at the beginning of the main.py file. You can see them in the help, using ```python main.py -h```.

The start and stop point can be choosen from the following image:

<p align="center">
<img src="./doc/img/town_positions.jpg" alt="Town Position" width="700"/>
</p>

## How does it work? 
The behavioral planner implements the following FSM.

<p align="center">
<img src="./doc/img/fsm.jpg" alt="fsm" width="700"/>
</p>

The check for traffic lights, pedestrians, and vehicle is made using CARLA perfect measures. You can see the areas in witch we
check for pedestrians and vehicles, and all that who intersects those areas, in the "Geometry Help" window.

<p align="center">
<img src="./doc/img/geometry_help.png" alt="geometry help" width="700"/>
</p>

The areas where the presence of people and pedestrians are checked and the stop lines of the traffic lights that intersect the path in the direction seen by the vehicle are represented.


## Scenatios of interest
For each scenario we report the functioning and the command line to test it.

* The vehicle stops at the traffic lights even if it's green because there's a pedestrian crossing the street. The vehicle passes the traffic lights even if it's yellow because it has already committed the intersection.
<p align="center">
<img src="./doc/gif/29_151_1_1000_1_1.gif" alt="scene 29_151" width="700"/>
</p>

        python main.py -ss 29,151 -nv 1 -np 1000 -sv 1 -sp 1

* Pedestrians cross. Still pedestrian crossing. We stop behind stationary vehicles and pedestrians make a mess of cars and one breaks like the exorcist.

        python .\main.py -ss 127,12 -np 1000 -nv 1000

* Pedestrian come too late.

        python .\main.py -ss 127,12 -np 100 -nv 100

* Stop at a Red Light

        python .\main.py -ss 107,120 -np 100 -nv 100