# CARLA Behavioural Planner
This is a university project for the self-driving vehicle course.
Below is information on how to launch the code and some demos.

<p align="center"><img src="./doc/gif/123-46-150-300-1-1-gif.gif" alt="Presentation scene" width="720" /></p>
 
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

### Test Planning and results
The following scenarios were analyzed:



### Test summary
A summary of the tests described above is as follows.

```mermaid
pie showData
    title Test summary
    "Passed" : 7
    "Passed with minor issues" : 5
    "Not Passed" : 2
```

The types of errors can instead be understood in the following graph.

```mermaid
pie showData
    title Error Tipology
    "Useless deceleration" : 2
    "Strange data from Carla" : 5
    "Other" : 0
```

### Test Analysis
Below you can see the execution of this test and the arguments to run that.

 * Test CODE:
   * **Command to run**: 

            python main.py -ss -np -nv -sp -sv 
   * **Traffic Lights**:
   * **Pedestrian**:
   * **Vehicles**:
   * **Description**:
   <p align="center"><img src="./doc/gif/123-46-150-300-1-1-gif.gif" alt="NOME" /></p>

## How does it work? 
We implemented, starting from a basecode provided us by our University (University of Salerno), the behaviour planner. 

### ODD
Our behavioral planner is positioned in the following ODD:
 * Lateral control is provided, but it only takes care of following the right trajectory. The possibility of overtaking has not been implemented.
 * The longitudinal control, which takes care of maintaining speed.
 * OEDR: the vehicle is capable of handling dangerous situations, such as pedestrians crossing the road.
 * The vehicle is not able to autonomously manage dangerous situations that require a response to the moral experiment. For example, the vehicle is unable to choose whether the lesser evil is crashing into the car in front of it or having several pedestrians injure themselves on it (pedestrian suicide in Carla).
 * The vehicle is limited to a situation where the map waypoints are known. Furthermore, data on the position and status of other vehicles, other pedestrians and traffic lights must also be known. It can be assumed that you are on a smart road without non-intelligent vehicles.
 * It is assumed that in most cases the other agents comply with the rules of the road, for example that the vehicles all walk in the right direction and in the right lane, and that they do not crash into other vehicles (although this does not always occur in simulations ).

### Architecture
The architecture of the behaviour planner is located between the Mission Planner and the Local Planner. You can see them in the following figure.
<p align="center">
<img src="./doc/img/behavioural_architecture.jpg" alt="fsm" width="700"/>
</p>

We can see that we receive the path waypoints from the Carla Mission Planner, and the perfect data from Carla as if we were on a smart road where vehicles, pedestrians and traffic lights share position and status. 
* The module "*Path Analyzer*" creates the path considering the wayponts to be considered based on the lookahead, directly proportional to the speed.
* The modules "*Pedestrians*", "*Vehicles*" and "*Traffic Lights*" take as input the calculated route and the data of carla and check if some of these objects are present in the route and if they are relevant for the scene in question. As regards vehicles, for example, only those present in the same lane as the car are considered lead vehicles. As far as pedestrians are concerned, both those on the road and those on the pavement are considered obstacles and will arrive on the road in front of the car in at most double the time it will take the car to get to that point.
* The waypoints and the information provided by these other modules are taken as input by the *FSM*, which calculates the outputs to be provided to the local planner, that is the desired state and the eventual vehicle to follow by adapting the trajectory.

The FSM module can be seen in the following figure.
<p align="center">
<img src="./doc/img/fsm.jpg" alt="fsm" width="700"/>
</p>

Three states are defined in the FSM, with the following meaning:
 * **Follow Lane**: In this state the car follows the trajectory in the absence of events that involve having to stop in front of a pedestrian, a vehicle or a traffic light.
 * **Decelerate to Stop**: In this state, a pedestrian has been detected who is or will be on the road for which it is necessary to stop, or a traffic light that has been red or yellow (but without the car having yet engaged the intersection), or the lead car is stopped in front of the car. This is needed to decelerate and stop.

It can be seen that the transitions cover all combinations of the following possibilities, giving priority to the closest obstacle:
 * **Trafficlight**: The favorable occasions to pass the traffic light are those in which it is green or yellow but only if the vehicle has already engaged the intersection. The unfavorable occasions are all the others, that is with a yellow traffic light and without having occupied the intersection or with a red light.
 * **Vehicle**: If there is a vehicle in the path of the car, it must adapt its speed to maintain the gap. It that vehicle is stopped, the car need to stop.
 * **Pedestrian**: If a pedestrian is in the middle of the road in the lane of the car or if it is expected that it will arrive on the trajectory in at most twice the time it will take the car to arrive at the same point, the car must stop before the pedestrian and until this will not leave the lane.

## Other changes
Compared to the code provided, other changes have been made that are not dependent on the assignment.

 * The controller for the lateral control has been replaced with the Stanley controller.
 * The longitudinal controller has been re-calibrated
 * Top speed has been increased from 18km/h to 40km/h on the straight and from 6.3km/h to 18km/h
 * Two windows have been added, which print information about the current information that the behavioral planner has
