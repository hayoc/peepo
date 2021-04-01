<img src="https://i.imgur.com/asqHaeo.png" alt="peepo" align="left" height="150" width="150"/>

## peepo: an attempt at artificial general intelligence

> [N]othing seems more possible to me than that people some day will come to the definite opinion that there is no copy in the [...] nervous system which corresponds to a particular thought, or a particular idea, or memory. (Ludwig Wittgenstein, Writings on the Philosophy of Psychology, 1948)

For more info [peepo.ai](https://peepo.ai) or follow our project board [taiga](https://tree.taiga.io/project/hayoc-peepo/backlog)

This version is the prototype of peepo. A version with the aim of performance, written in C++, can be found here: https://github.com/hayoc/peepo

## Installation
```bash
python setup.py install
```

## Usage
The repository consists of multiple experiments which have been implemented.

- `peepo\bot` 
    * Requirements: [(LEGO MINDSTORMS EV3)](https://www.lego.com/en-us/mindstorms/products/mindstorms-ev3-31313) and [(ev3dev)](https://www.ev3dev.org) 
- `peepo\playground`
    * `survival` contains a _peepo_ implementation searching for food in a virtual world.  
    Video: https://www.youtube.com/watch?v=aOjK5MW6E-U
    * `wandering` contains a _peepo_ implementation avoiding obstacles.  
    Video: https://www.youtube.com/watch?v=kMOr368zJ-g
    * `hawk` contains a _peepo_ implementation capturing a prey using target interception.  
    Video: https://www.youtube.com/watch?v=WT0A4p6W8os 
    
## License
[MIT](https://choosealicense.com/licenses/mit/)
