A tree structure that models the switching of different tasks in an autonomous agent.

There are a number of key benefits a behavior tree has over a [[Finite State Machine]].

**1. They are inherently hierarchical:** a bigger tree can be composed of smaller sub-branches of reusable behaviours.
**2. Their graphical representation has a semantic meaning:** its readable...? someone said this, its kinda readable i guess once you get use to it.

![[Pasted image 20260209140803.png]]

Fundamentally, the behaviour tree functions off a tick. This tick is periodic and travels from the root of the tree down to the behaviour to execute.

There are different types of nodes in a behaviour tree. 
- **Control Node** (many children) ticks a child based on the result of its siblings or its own state. These nodes control the flow of the ticks (ie. **Fallback Node**, or **Sequence Node**)
- **Decorator Node** (1 child) alters the result of its child or tick it multiple times
- **Condition Node** (no child) does not alter the system, return SUCCESS, FAILURE but not RUNNING
- **Action Node** (no child) a node that does something

