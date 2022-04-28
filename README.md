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


## Scenarios of interest
For each scenario we report the functioning and the command line to test it.

### Quantitative Analysis

#### Semaphor
The following scenarios were analyzed:

| start | stop | state | note | passed |
|-------|------|-------|------|--------|
| 121   | 96   |       |      |        |
| 47    | 51   |       |      |        |
| 51    | 135  |       |      |        |
| 57    | 135  |       |      |        |
| 11    | 24   |       |      |        |
| 57    | 52   |       |      |        |
| 141   | 136  |       |      |        |
| 135   | 90   |       |      |        |
| 27    | 151  |       |      |        |

        python main.py -ss 121,96.45,51.51,135.57,135.11,24.57,52.141,136.135,90.27,151 -np 1 -nv 1 -sp 1 -sv 1

### Qualitative Analysis

* The vehicle stops at the traffic lights even if it's green because there's a pedestrian crossing the street. The vehicle passes the traffic lights even if it's yellow because it has already committed the intersection.
<p align="center">
<img src="./doc/gif/29_151_1000_1_1_1.gif" alt="scene 29_151" width="700"/>
</p>

        python main.py -ss 29,151 -np 1000 -nv 1 -sv 1 -sp 1

* The vehicle stops for the crossing of some pedestrians, after which it restarts to stop behind some stationary vehicles. Finally, some pedestrians throw themselves on the stationary car and a motorcycle creates a rear-end collision with our still stationary car.

        python .\main.py -ss 127,12 -np 1000 -nv 1000

* A pedestrian jumps into the road too close to the car, which doesn't have time to brake.

        python .\main.py -ss 127,12 -np 100 -nv 100


#### Traffic Lights

* The vehicle slows down and stops at a red light, then starts again when the green light turns on.
        
        python .\main.py -ss 11,19 -np 1 -nv 1 -sp 1 -sv 1

* The vehicle decelerates at a red light and, during deceleration, the light turns green and the vehicle accelerates again.

        python .\main.py -ss 8,19 -np 1 -nv 1 -sp 1 -sv 1

* The vehicle at a green light

        python .\main.py -ss 47,51 -np 1 -nv 1 -sp 1 -sv 1

* Vehicle is about to pass the traffic light when it turns yellow, so it continues because it has already engaged the intersection.

        python .\main.py -ss 62,67 -np 1 -nv 1 -sp 1 -sv 1

#### Pedestrians

* Pedestrian throws himself in the middle of the road, so we slow down so as not to run over him.
        
        python .\main.py -ss 4,11 -np 200 -nv 1 -sp 1 -sv 1

* The vehicle, when another vehicle enters its lookahead, adjusts the speed to maintain the safety fap. 
When the vehicle stops at the traffic lights we queue even though the traffic lights are not yet in the lookahead of our vehicle. 
When the vehicle in front of us restarts, our car also restarts. At the traffic lights, at the yellow click, the vehicle passes anyway because it has already engaged the intersection. 
The vehicle collides with another vehicle when cornering due to imperfect control, which is not the purpose of this exercise.
        
        python .\main.py -ss 0,24 -np 1 -nv 200 -sp 1 -sv 1

