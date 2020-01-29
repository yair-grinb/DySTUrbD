This directory includes a set of synthetic files demonstrating the kind of inputs expected by the model:
1. An agents file (dummy_agents.csv) including the following columns: 
   1. ID
   2. Household ID
   3. Disability (0 = no physical disability, 1 = has disability)
   4. Employment status (0 = unemployed, 1 = employed)
   5. Age group (1 = child, 2 = adult, 3 = elderly)
   6. Employment location (0 = outside of the world, 1 = locally employed)
2. A households file (dummy_households.csv) including the following columns:
   1.  ID
   2. Place of residence - building ID
   3. Monthly income
   4. Car ownership (0 = does not own a car, 1 = owns a car)
3. A buildings file (dummy_bldgs.csv) including the following columns:
   1. ID
   2. Land use (0 = unoccupied, 1 = residential, 3 = commercial, 4 = public use)
   3. Number of floors
   4. Zone ID
   5. Floorspace volume
   6. X coordinate
   7. Y coordinate
4. A zones file (dummy_zones.csv) including the following columns:
   1. ID
   2. Average out-migration volume (not used in the model)
   3. Chance for out-migration
   4. Average housing price per sqm
5. A roads file (dummy_roads.csv) including the following columns:
   1. ID
   2. X coordinate for start point
   3. Y coordinate for start point
   4. X coordinate for end point
   5. Y coordinate for end point
   6. Length
