Context: [[Sampled-Data Control Systems]]

# Choosing a Sampling time
You can choose a sampling time by analyzing the frequency response of the plant (use a bode plot).

>[!info] Remember! A bode plot is equivalent to giving the plant a higher and higher "shake" input. If you shake the input so much, the plant wont have the opportunity to respond fast enough, and at some point it will just stop moving all together.

Here's what we can deduce:
-  Get the bode plot of the plant
	![[Pasted image 20251027122655.png]]
	- $\omega_{bw}$ can represent the smallest timescale we can use the plant in, any faster and the system devolves
		- To locate find $\omega_{bw}$ when $|P(j\omega_{bw})|=-3dB$
		- Or $\omega_{bw}=max_{\text{poles}\;p_{i}\text{in}\;P(s)}|\mathrm{Re}(p_{i})|$
			- of all the poles of the plant, find the one with the highest real part
- **Set the sampling frequency accordingly,** we are denoted the frequency of the sample time as $\omega_{s}=\frac{2\pi}{T}$
> [!info] important! ^^
- We then want to also account for the frequency of the reference signal we desire, as well as the frequency of the disturbance (denoted as $\omega_{0}$)
	- And we ideally desire $\omega_{s} >max(\omega_{bw},\omega_{0})$
		- But the higher the frequency, the more expensive 
		- Also may not be feasible because hardware limitations
		- Also may not be feasible because of computation time
		- Also may not be feasible because we reach the limits of numerical precision (float8 not precise enough!!)

Keeping in mind all of that, **we have a couple of options**:
1. **Choose a small T if feasible**
	- **Rule of thumb:** $\omega_{s}>(5,10)\cdot max{(\omega_{bw},\omega_{0})}$
2. **Add a continuous-time low pass filter**
	![[Pasted image 20251027124037.png]]
	- Filter out frequencies higher than $\frac{\omega_{s}}{2}$ by designing a filter with $\omega_{filter}< \frac{\omega_{s}}{2}$
	- we can't set a $\omega_{filter}$ too low otherwise we will get a delayed signal (called phase lag because we are thinking in the frequency domain most of the time)
		- Rule of thumb here is $\omega_{filter}>\omega_{bw}$


