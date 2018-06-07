# kalman-folding
Playing with Kalman folding à la Brian Beckman [^kf-1] [^kf-2] [^kf-4].

***Warning:*** *this is a work in progress.*

***NB:*** I am extending this technique to control systems in general, and nonlinear model predictive control in particular. Check back soon for repositories exhibiting that research.

(To download Dr. Beckman's papers, and see others in the series, [click here](http://vixra.org/author/brian_beckman).)

## Organisation

Everything herein is written as unit-test (using [Catch2](https://github.com/catchorg/Catch2)), so look in `/test` for the goods. In retrospect, I question this choice.


Assuming this readme is up-to-date, the test directory should contain:
```
└── test
    ├── catch.cpp
    ├── CMakeLists.txt
    ├── utest-falling_object.cpp
    ├── utest-kalman_folding_4-frp.cpp
    └── utest-linear_least_squares.cpp
```
In [^kf-1], Dr. Beckman introduces the static Kalman filter in a series of four preludes, the fourth being an implementation of a static Kalman filter. (Static meaning that model states do not vary with the independent variable). Those preludes are reproduced in `utest-linear_least_squares.cpp`.

In [^kf-2], Dr. Beckman generalizes to the non-static case where the model includes a control input term in addition to the drift term. Beckman's exhibition centres on a textbook example from Zarchan and Musoff[^Z&M]. This is reproduced in `utest-falling_object.cpp`.

Finally, in [^kf-4] Dr. Beckman extends the method to streams and observables.  Since a key advantage of these lines of abstractions is that the Kalman filtering behaviour is decoupled from the data structures used to organize input and output of the filter, I didn't take care to reproduce this paper faithfully.  Instead, I am most interested in the push-based (observable) case. I opted to use [Sodium](https://github.com/SodiumFRP/sodium-cxx), a library implementing Conal Elliott's Functional Reactive Programming (FRP). That work is found in `utest-kalman_folding_4-frp.cpp`.

These files are meant to be self-documenting. If I've been unclear anywhere, and you feel something deserves explanation, even when read alongisde Beckman's papers, please file an issue.

## Dependencies:
 * CMake
 * [Catch2](https://github.com/catchorg/Catch2)
 * [Eigen3](ihttp://eigen.tuxfamily.org/index.php?title=Main_Page) (Debian/Ubuntu: `sudo apt install libeigen3-dev`)
 * [Boost.Hana](https://www.boost.org/doc/libs/1_61_0/libs/hana/doc/html/index.html) (only for Currying, I'll probably eliminate this dependency if anyone asks.)
 * [Range-v3](https://github.com/ericniebler/range-v3)

[^kf-1]: Brian Beckman, Kalman Folding-Part 1. (2016)
[^kf-2]: Brian Beckman, Kalman Folding 2: Tracking and System Dynamics. (2016)
[^kf-4]: Brian Beckman, Kalman Folding 4: Streams and Observables. (2016)
[^Z&M]: Zarchan and Musoff, Fundamentals of Kalman Filtering: A Practical Approach. 4th Ed. Ch 4.

