# Genetic-Algorithm-Driven-Molecular-Design-in-Latent-Space
Molecular Optimization Using Genetic Algorithm and VAE

Overview
This repository contains a Python script for generating and optimizing molecular structures using a genetic algorithm (GA) integrated within the latent space of a Variational Autoencoder (VAE). The VAE model is based on the "Junction Tree Variational Autoencoder for Molecular Graph Generation" by Wengong Jin et al. This innovative approach leverages the GA to explore and optimize molecular structures in the latent space defined by the VAE.

Concept
The project demonstrates a proof of concept where a genetic algorithm optimizes molecule generation within the latent space of a pre-trained VAE. The initial molecule is generated randomly but can be substituted with a specific molecule to seed the initial population for the GA. This approach allows for the exploration of novel molecular structures that are not only diverse but also optimized for specific properties such as drug-likeness and biological activity.

Features
- Molecular Generation: Start with a random or specified molecule to seed the genetic algorithm.
- Latent Space Exploration: Utilizes the latent space of a Junction Tree Variational Autoencoder to ensure that generated molecules are chemically valid.
- Property Optimization: Incorporates multiple molecular properties into the fitness function of the GA, including compliance with Lipinski's Rule of Five.
- Tanimoto Similarity: Integrates Tanimoto similarity to the initial molecule in the fitness calculation, enhancing the generation of molecules similar to a target structure.

 Usage
To run the molecular optimization script, ensure you have Python installed along with necessary libraries including RDKit, PyTorch, and DEAP. Adjust the parameters in the script to fit your specific requirements:
- `tanimoto_weight`: Weight of the Tanimoto similarity score in the fitness function.
- `ngen` (number of generations): Total generations for GA to run.
- `cxpb` (crossover probability): Probability with which two individuals are crossed.
- `mutpb` (mutation probability): Probability of mutating an individual.
- `toolbox.population(n=50)`: Size of the population in each generation.

These parameters can be tuned to balance exploration and exploitation, adapt to different molecular targets, or adjust the computational complexity of the run.

Potential Improvements
This proof of concept is open for further development and optimization:
- Parameter Tuning: Experimentation with GA parameters for better convergence and diversity.
- Integration with Experimental Data: Incorporating feedback from synthetic and biological testing to refine fitness functions.
- Scalability: Enhancing the script to handle larger populations and more complex molecular properties efficiently.

 References
This project uses the VAE architecture proposed by Wengong Jin et al. in their work "Junction Tree Variational Autoencoder for Molecular Graph Generation".
