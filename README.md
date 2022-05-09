# CARLA-behavioural-planner

## How to run
To run Carla Server whit this script, you need the following settings:

    .\CarlaUE4.exe /Game/Maps/Town01 -carla-server -windowed -benchmark -fps=30

To run the script consider the following arguments:

    python main.py -ss START1,STOP1.START2,STOP2.[...] -nv 10,100,[...] -np 10,100.[...] -sv 1,2,67,[...] -sp 1,4,77,[...]

where
 * ```ss``` is an argument which require the stary waypoint and the stop waypoint, separated by a comma. 
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

**NOTE**: You must install the requirements in *requirements.txt*.

## Scenarios of interest
For each scenario we report the functioning and the command line to test it.
The following scenarios were analyzed:

| start | stop | np   | nv   | sp | sv | semaphore | pedestrians | vehicles | passed | note                                                                                                                                                                                                          |
|-------|------|------|------|----|----|-----------|-------------|----------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 121   | 96   | 1    | 1    | 1  | 1  | green     | no          | no       | yes    | The semaphore is green, so the car pass without slow down.                                                                                                                                                    |
| 47    | 51   | 1    | 1    | 1  | 1  | green     | no          | no       | yes    | The semaphore is green, so the car pass without slow down.                                                                                                                                                    |
| 51    | 135  | 1    | 1    | 1  | 1  | green     | no          | no       | yes    | The semaphore is green, so the car pass without slow down.                                                                                                                                                    |
| 57    | 135  | 1    | 1    | 1  | 1  | red       | no          | no       | yes    | The semaphore is red, so the car stop. When the semaphore turn green, the vehicle restart.                                                                                                                    |
| 11    | 24   | 1    | 1    | 1  | 1  | red       | no          | no       | yes    | The semaphore is red, so the car stop. When the semaphore turn green, the vehicle restart.                                                                                                                    |
| 57    | 52   | 1    | 1    | 1  | 1  | red       | no          | no       | yes    | The semaphore is red, so the car stop. When the semaphore turn green, the vehicle restart.                                                                                                                    |
| 141   | 136  | 1    | 1    | 1  | 1  | green     | no          | no       | yes    | The semaphore is green, so the car pass without slow down.                                                                                                                                                    |
| 135   | 90   | 1    | 1    | 1  | 1  | red       | no          | no       | yes    | The semaphore is red, so the car stop. When the semaphore turn green, the vehicle restart.                                                                                                                    |
| 27    | 151  | 1    | 1    | 1  | 1  | red       | no          | no       | yes    | The semaphore is red, so the car stop. When the semaphore turn green, the vehicle restart.                                                                                                                    |
| 29    | 151  | 1000 | 1    | 1  | 1  | red       | 2           | no       | yes    | The vehicle is stopped at the red trafficlights. When the trafficlight turns red, a pedestrian start crossing the road and the vehicle stop. When the pedestrian stop to cross the road, the vehicle restart. |
| 127   | 12   | 1000 | 1000 | 1  | 1  | no        | 5           | 2        | yes    | A pedestrian start crossing the road, but we can pass that point before that enter the road, so the car pass. After we stop behind a car stopped because some pedestrians are crossing the road.              |
| 127   | 12   | 100  | 100  | 1  | 1  | red, red  | 3           | 1        | yes    | Some pedestrian, during this simulation, are about to cross the way, but we can pass that zone before they come in the road, so the car does not stop. After the car stop behind some stopped cars.           |


Below you can see the execution of this test and the arguments to run that.

<p align="center">
<img src="./doc/gif/test_trafficlights-gif.gif" alt="scene 29_151" width="700"/>
</p>

        python main.py -ss 121,96.45,51.51,135.57,135.11,24.57,52.141,136.135,90.27,151 -np 1 -nv 1 -sp 1 -sv 1

<p align="center">
<img src="./doc/gif/29-151-1000-1-1-1-gif.gif" alt="scene 29_151" width="700"/>
</p>

        python main.py -ss 29,151 -np 1000 -nv 1 -sv 1 -sp 1

<p align="center">
<img src="./doc/gif/89-133-1000-1000-1-0-gif.gif" alt="scene 127_12" width="700"/>
</p>

        python .\main.py -ss 127,12 -np 1000 -nv 1000

<p align="center">
<img src="./doc/gif/127-12-100-100-0-0-gif.gif" alt="scene 127_12" width="700"/>
</p>

        python .\main.py -ss 127,12 -np 100 -nv 100

<p align="center">
<img src="./doc/gif/4-11-200-1-1-1-gif.gif" alt="scene 4_11" width="700"/>
</p>

        python .\main.py -ss 4,11 -np 200 -nv 1 -sp 1 -sv 1

## How does it work? 
We implemented, starting from a basecode provided us by our University, the behaviour planner. The architecture of the behaviour planner is located between the Mission Planner and the Local Planner. You can see them in the following figure.
<p align="center">
<img src="./doc/img/behavioural_architecture.jpg" alt="fsm" width="700"/>
</p>

The FSM module can be seen in the following figure.
<p align="center">
<img src="./doc/img/fsm.jpg" alt="fsm" width="700"/>
</p>

The check for traffic lights, pedestrians, and vehicle is made using CARLA perfect measures. You can see the areas in witch we
check for pedestrians and vehicles, and all that who intersects those areas, in the "Geometry Help" window.

<p align="center">
<img src="./doc/img/geometry_help.png" alt="geometry help" width="700"/>
</p>

The areas where the presence of people and pedestrians are checked and the stop lines of the traffic lights that intersect the path in the direction seen by the vehicle are represented.

