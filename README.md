# CARLA Behavioural Planner
This is a university project for the self-driving vehicle course.
Below is information on how to launch the code and some demos.

## Demo Videos
[![Traffic Lights demo](http://img.youtube.com/vi/5L9QwT_yl1k/0.jpg)](http://www.youtube.com/watch?v=5L9QwT_yl1k "Traffic Lights Demo")

[![Vehicles demo](http://img.youtube.com/vi/GhraqP6mP30/0.jpg)](http://www.youtube.com/watch?v=GhraqP6mP30 "Vehicles Demo")

[![Pedestrians demo](http://img.youtube.com/vi/UIo7Ag_J8qE/0.jpg)](http://www.youtube.com/watch?v=UIo7Ag_J8qE "Pedestrians Demo")

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

### Test Analysis
Below you can see the execution of this test and the arguments to run that.

 * Test: 116-140-450-1-1-1
   * **Command to run**: 

            python main.py -ss 116,140 -np 450 -nv 1 -sp 1 -sv 1
   * **Traffic Lights**: 1 yellow
   * **Pedestrian**: 4
   * **Vehicles**: 0
   * **Description**: A pedestrian crossing the way, so the car stops before the pedestrian and restart when he’s passed.
                      A yellow traffic light, so the car decelerates to stop before the traffic light that turns red.
                      A pedestrian on the lane, so the car decelerates while the pedestrian leaves the street. 
                      Two pedestrians that cross the street, so the car stops until they leave the street.

   <p align="center"><img src="./doc/gif/116_140_1_1_450_1.gif" alt="NOME" /></p>

 * Test: 116-140-300-150-1-1
   * **Command to run**: 

            python main.py -ss 116,140 -np 300 -nv 150 -sp 1 -sv 1
   * **Traffic Lights**: 1 red
   * **Pedestrian**: 3
   * **Vehicles**: 1 motorcycle
   * **Description**: A motorcycle in front of our vehicle, so the car follows it. The motorcycle stops at a red traffic light, so the car stops behind it until it restarts at the green light. The motorcycle stops to let a pedestrian cross, so the car stops behind it until it restarts. The motorcycle stops behind two cars stopped at a red traffic light, so the car stops. The motorcycle restarts but the car stay stopped to let two pedestrians cross. A pedestrian suicide on the car. A detection error is verified, so the car doesn’t see the motorcycle and collides with it

   <p align="center"><img src="./doc/gif/116_140_300_150_1_1.gif" alt="NOME" /></p>

 * Test: 89-119-150-150-3-1
   * **Command to run**: 

            python main.py -ss 89,119 -np 150 -nv 150 -sp 3 -sv 1
   * **Traffic Lights**: 1 yellow
   * **Pedestrian**: 0
   * **Vehicles**: 1 car
   * **Description**: A yellow traffic light, wo the car decelerates to stop until it turns green. A car is stopped at a red traffic light behind other cars, so our vehicle stops until they restart

   <p align="center"><img src="./doc/gif/89_119_150_150_3_1.gif" alt="NOME" /></p>

 * Test: 130-12-100-100-0-0
   * **Command to run**: 

            python main.py -ss 130,12 -np 100 -nv 100 -sp 0 -sv 0
   * **Traffic Lights**: 2 red
   * **Pedestrian**: 1
   * **Vehicles**: 1 car
   * **Description**: A car in front of our vehicle, so we follow it. A pedestrian crosses the street, so we stop to let him pass but he suicides on our vehicle. Our vehicle restarts and then it stops behind a car stopped at a red traffic light until it turns green. A red traffic light is present, so the vehicle stops

   <p align="center"><img src="./doc/gif/130_12_100_0_100_0.gif" alt="NOME" /></p>

 * Test: 123-46-150-150-3-1
   * **Command to run**: 

            python main.py -ss 123,46 -np 150 -nv 150 -sp 3 -sv 1
   * **Traffic Lights**: 4 red
   * **Pedestrian**: 0
   * **Vehicles**: 1 bike, 1 car
   * **Description**: A bike in front of our vehicle, so we follow it.The bike stops at 2 red traffic light, so we stop behind it. A car in front of our vehicle, so we follow it. The car stops at two red traffic light, so we stop behind it

   <p align="center"><img src="./doc/gif/123_46_150_3_150_1.gif" alt="NOME" /></p>

 * Test: 26-15-1-1-1-1
   * **Command to run**: 

            python main.py -ss 26,15 -np 1 -nv 1 -sp 1 -sv 1
   * **Traffic Lights**: 1 green, 1 yellow
   * **Pedestrian**: 0
   * **Vehicles**: 0
   * **Description**: •	A green traffic light is present, so the car continues on its path. A traffic light turns yellow when the car has already joined the intersection, so the car continue on its path

   <p align="center"><img src="./doc/gif/26_15_1_1_1_1.gif" alt="NOME" /></p>

 * Test:125-28-1-1-1-1
   * **Command to run**: 

            python main.py -ss 125,28 -np 1 -nv 1 -sp 1 -sv 1
   * **Traffic Lights**: 1 red
   * **Pedestrian**: 0
   * **Vehicles**: 0
   * **Description**: A red traffic light, so the car stops until it turns green and the restarts

   <p align="center"><img src="./doc/gif/125_28_1_1_1_1.gif" alt="NOME" /></p>

### Test summary
A summary of the tests described above is as follows.

```mermaid
pie showData
    title Test summary
    "Passed" : 10
    "Not Passed" : 2
```

The real quantity of error can be seen in the following analysis.

```mermaid
pie showData
    title Situation managed
    "Passed" : 63
    "Non Passed" : 2
```

## How does it work? 
We implemented, starting from a basecode provided us by our University (University of Salerno), the behaviour planner. 

### ODD
Our behavioral planner is positioned in the following ODD:
 * Lateral control is provided, but it only takes care of following the right trajectory. The possibility of overtaking has not been implemented.
 * The longitudinal control, which takes care of maintaining speed.
 * OEDR: the vehicle is capable of handling dangerous situations, such as pedestrians crossing the road.
 * The vehicle is not able to autonomously manage dangerous situations that require a response to the moral experiment. For example, the vehicle is unable to choose whether the lesser evil is crashing into the car in front of it or having several pedestrians injure themselves on it (pedestrian suicide in Carla).
 * The vehicle is limited to a situation where the map waypoints are known. 
 * It is assumed that depth data and semantic segmentation are “perfect”. 
 * It is assumed that in most cases the other agents comply with the rules of the road, for example that the vehicles all walk in the right direction and in the right lane, and that they do not crash into other vehicles (although this does not always occur in simulations ).

### Architecture
The architecture of the behaviour planner is located between the Mission Planner and the Local Planner. You can see them in the following figure.
<p align="center">
<img src="./doc/img/behavioural_architecture.jpg" alt="fsm" width="700"/>
</p>

We can see that we receive the path waypoints from the Carla Mission Planner, and the perfect data from Carla as if we were on a smart road where vehicles, pedestrians and traffic lights share position and status. 
* The module "*Path Analyzer*" creates the path considering the wayponts to be considered based on the lookahead, directly proportional to the speed.
* The module Traffic Lights take as input the calculated route and the Carla’s data and check if traffic light is present in the route and its state.  
* The modules Pedestrians and Vehicles takes as input the calculated route, the position of the vehicle computed from the camera’s image, and “perfect” data for speed and orientation. These modules check if some of these objects are present in the route and if they are relevant for the specific scene. As regards vehicles, for example, only those present in the same lane as the car are considered lead vehicles. Meanwhile detected pedestrians are considered obstacles either they are on the road either they are on the sidewalk and will arrive on the road in front of the car in at most double the time it will take the car to get to that point. 
* The waypoints and the information provided by these other modules are taken as input by the FSM, which calculates the outputs to be provided to the local planner, which is the desired state and the eventual vehicle to follow by adapting the trajectory. 

The module Local Planner has been modified to implement a Stanley controller tuned for handling the lateral control. 

The FSM module can be seen in the following figure.
<p align="center">
<img src="./doc/img/fsm.jpg" alt="fsm" width="700"/>
</p>

Three states are defined in the FSM, with the following meaning:
 * **Follow Lane**: In this state the car follows the trajectory in the absence of events that involve having to stop in front of a pedestrian, a vehicle or a traffic light.
 * **Decelerate to Stop**: In this state, a pedestrian has been detected who is or will be on the road for which it is necessary to stop, or a traffic light that has been red or yellow (but without the car having yet engaged the intersection), or the lead car is stopped in front of the car. This is needed to decelerate and stop.

It can be seen that the transitions cover all combinations of the following possibilities, giving priority to the closest obstacle:
 * **Trafficlight**: The favorable occasions to pass the traffic light are those in which it is green or yellow but only if the vehicle has already engaged the intersection. The unfavorable occasions are all the others, that is with a yellow traffic light and without having occupied the intersect