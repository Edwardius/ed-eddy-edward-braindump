# Sampled-Data (Digital) Control Systems

![[Pasted image 20251027120500.png]]

Here we have a controller that requires Digital/Analog conversions.

- $D[z]$ discrete-time controller
- $r,d,u,y$ are all in **continuous time**
- A/D is the sampler (analog to digital converter)
	- samples at discrete intervals $kT, k\in\mathbb{Z}\geq_{}0$
- T is the **sampling time** of the system
- D/A is the zero order hold (Digital to Analog converter) 
	- holds $u[k]$ for $t\in[kT,(k+1)T]\;\forall k\in\mathbb{Z}\geq_{0}$

![[Pasted image 20251027120952.png]]
![[Pasted image 20251027121007.png]]

>[!info] This system in NOT LTI (Linear Time Invariant)! Thats becuase the signal can change if you shift the time off a bit (will sample and hold completely different signals)

**As a result, what do we do?**
- Attempt to choose a sampling time of the control architecture in such a way to make the system as LTI as possible
- Utilize LTI tools anyways
	- Approximating continuous time controllers with discrete controllers
	- or by directly designing a discrete controller
- Analyze and simulate the results on the simulated true system (accounting for the samplers and holders)

# Example
![[Pasted image 20251027121925.png]]
1. Our sampler gets worse and worse the higher the frequency of the input signal.
![[Pasted image 20251027122000.png]]
2. We can cause instability if our sampling time is too long
![[Pasted image 20251027122214.png]]
