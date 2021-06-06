# DySTUrbD
Dynamic Simulation Tool for Urban Disasters (DySTUrbD) - an agent-based simulation of long-term urban recovery from disasters

This agent-based simulation model was developed as part of a research project at the Department of Geography, the Hebrew University of Jerusalem, aimed at developing an understanding of urban recovery processes in the wake of a disaster. The model simulates the urban system as consisting of three main sub-systems: the housing market, the land-use market, and the labor market. Agents represent individuals and are aggregated into households. The environment includes buildings, roads, jobs, and zones (representing neighborhoods, census tracts, or local housing markets). Environmental entities are sensitive to their surroundings and may change attributes accordingly. The model introduces into the system a physical shock representing an earthquake which demolishes buildings, hence directly affecting land-uses and residents and indirectly future rounds of decisions, and simulates the emergence of urban patterns following it.

The epidemiological model:
1. Contagious - the script which contains the agents' routine calculations, the epidemiological calculations and the calls to create the data and the social network. there are more versions of this script, contagious3 is most up to date.
2. create_random_data - a script which takes the raw agents & buildings data and makes manipulations over it in order to create the final data which is used for calculations.
3. communities - the creation of the social network.
4. parameters - the model's pararmeters.
5. create_figures - figures which are created from the results of the simulations.

The model includes the following files:
1. main.py - the script which calls on the functions for initiating and running the model and for saving outputs.
2. model_parameters.py - this file details the parameters used to define dynamics in the model and their values.
3. global_variables.py - these are variables shared across model functions, e.g. the agents array.
4. create_world.py - a file containing the functions required for building the model's environment and classes based on input files found in the data directory.
5. ab_model.py - this file stores all the model's functions including the scheduling function (general_step) and the step models for every class of entities in the model.
6. auxilliary_functions.py - a collection of functions used both during the building of the environment and the operations of the model.
7. output_functions.py - a collection of functions used to store results.

The results of implementations of earlier versions of the model are detailed in the following publications:
1. Grinberger, A.Y., & Felsenstein, D. (2014). [Bouncing back or bouncing forward? Simulating urban resilience](https://www.icevirtuallibrary.com/doi/full/10.1680/udap.13.00021). Proceedings of the ICE: Urban Design and Planning, 167(3), 115-124. **Open Access**
2.	Grinberger, A.Y., Lichter, M., & Felsenstein, D. (2015). [Simulating urban resilience: disasters, dynamics and (synthetic) data](https://link.springer.com/chapter/10.1007/978-3-319-18368-8_6). In S. Geertman, J. Stillwell, J. Ferreira & R. Goodspeed (Eds.) Planning support systems and smart cities (pp. 99-119). Springer, Cham. **Open Access**
3. Lichter, M., Grinberger, A.Y., & Felsenstein, D. (2015). [Simulating and communicating outcomes in disaster management situations](https://www.mdpi.com/2220-9964/4/4/1827). ISPRS International Journal of Geo-Information, 4(4), 1827-1847. **Open Access**
4. Grinberger, A.Y., & Felsenstein, D. (2016). [Dynamic agent-based simulation of welfare effects of urban disaster](https://www.sciencedirect.com/science/article/pii/S0198971516300862). Computers, Environment and Urban Systems, 59, 129-141.
5.	Grinberger, A.Y., & Felsenstein, D. (2017). [A tale of two earthquakes: Dynamic agent-based simulation of urban resilience](https://www.taylorfrancis.com/books/e/9781315683621/chapters/10.4324/9781315683621-18). In J. Lombard, E. Stern & G. Clarke (Eds). Applied spatial modeling and planning (pp. 134-154). Routledge, Abingdon & New York. 
6.	Grinberger, A.Y., Lichter, M., & Felsennstein, D. (2017). [Dynamic agent based simulation of an urban disaster using synthetic big data](https://link.springer.com/chapter/10.1007/978-3-319-40902-3_20). In P. Thakuria, N. Tilahun & M. Zellner (Eds.) Seeing cities through big data: Theory methods and applications in urban informatics (pp. 349-382). Springer, Cham. **Open Access**
7. Grinberger, A.Y., & Samuels, P. (2018). [Modeling the labor market in the aftermath of a disaster: Two perspective](https://www.sciencedirect.com/science/article/pii/S2212420918306514). International Journal of Disaster Risk Reduction, 31, 419-434.
8.	Grinberger, A.Y., & Felsenstein, D. (2019). [Emerging urban dynamics and labor market change: An agent-based simulation of recovery from a disaster](https://www.elgaronline.com/view/edcoll/9781788970099/9781788970099.00019.xml). In K. Borsekova & P. Nijkamp (Eds.) Resilience and Urban Disasters (pp. 232-256). Edward Elgar, Cheltenham.
9. Felsenstein, D., & Grinberger, A. Y. (2020). [Cascading effects of a disaster on the labor market over the medium to long term](https://www.sciencedirect.com/science/article/pii/S2212420919309835?via%3Dihub). International Journal of Disaster Risk Reduction, in press.
