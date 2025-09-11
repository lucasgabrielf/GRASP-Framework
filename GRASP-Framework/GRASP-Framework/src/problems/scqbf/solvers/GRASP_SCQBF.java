 package problems.scqbf.solvers;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import metaheuristics.grasp.AbstractGRASP;
import problems.scqbf.SCQBF;
import solutions.Solution;

/**
 * Metaheuristic GRASP for the Set Cover Quadratic Binary Function (SC-QBF) problem.
 * This implementation uses a two-phase iterative approach:
 * 1. Construction: A feasible solution is built using a randomized subtractive heuristic.
 * 2. Local Search: The constructed solution is improved using a best-improvement
 * heuristic based on a 1-1 exchange neighborhood.
 * This process is repeated for a specified number of iterations, returning the best
 * solution found.
 *
 * @author adapted from ccavellucci, fusberti
 */
public class GRASP_SCQBF extends AbstractGRASP<Integer> {

    private final SCQBF scqbfObjectiveFunction;
    private final Random random = new Random();
    private final SearchStrategy searchStrategy;
    public enum ConstructionType { SUBTRACTIVE, RANDOM_GREEDY, SAMPLED_GREEDY }
    private final ConstructionType constructionType;

    /**
     * A helper class to store a candidate for removal and its associated cost.
     * This makes sorting and building the RCL easier.
     */
    private static class RemovalCandidate implements Comparable<RemovalCandidate> {
        public final Integer id;
        public final Double cost;

        public RemovalCandidate(Integer id, Double cost) {
            this.id = id;
            this.cost = cost;
        }

        @Override
        public int compareTo(RemovalCandidate other) {
            return this.cost.compareTo(other.cost);
        }
    }

    /**
     * Constructor for the GRASP_SCQBF class.
     *
     * @param alpha    
     * @param iterations
     * @param filename  
     * @throws IOException
     */
    public GRASP_SCQBF(Double alpha, Integer iterations, String filename, ConstructionType constructionType, SearchStrategy searchStrategy) throws IOException {
        super(new SCQBF(filename), alpha, iterations);
        this.scqbfObjectiveFunction = (SCQBF) this.ObjFunction;
        this.constructionType = constructionType;
        this.searchStrategy = searchStrategy;
    }

    /**
     * The main solver method. It runs the GRASP iterations, performing the
     * construction and local search phases, and returns the best solution found.
     */
    @Override
    public Solution<Integer> solve() {
        Solution<Integer> bestSol = null;

        for (int i = 0; i < iterations; i++) {
        	//Escolhe a heurística
        	Solution<Integer> currentSol;
        	switch (constructionType) {
            case RANDOM_GREEDY:
                currentSol = constructRandomPlusGreedy(5); // sorteia 5 elementos
                break;
            case SAMPLED_GREEDY:
                currentSol = constructSampledGreedy(10);   //avalia 10 elementos sorteados
                break;
            default:
                currentSol = runSingleSubtractiveIteration();
        	}
        	// Phase 2: Local Search
            currentSol = localSearch(currentSol, searchStrategy);
            if (bestSol == null || currentSol.cost > bestSol.cost) {
                bestSol = new Solution<>(currentSol);
            }
        }
        return bestSol;
    }

    /**
     * The construction phase of the GRASP iteration.
     * Starts with a full solution and prunes it based on the RCL mechanism.
     *
     * @return A feasible solution.
     */
    private Solution<Integer> runSingleSubtractiveIteration() {
        // 1. Start with the full solution.
        Solution<Integer> currentSolution = createFullSolution();

        while (true) {
            // 2. Build a list of all candidates that can be feasibly removed.
            List<RemovalCandidate> removableCandidates = new ArrayList<>();
            for (Integer candOut : currentSolution) {
                if (scqbfObjectiveFunction.isFeasibleAfterRemoval(candOut, currentSolution)) {
                    double cost = scqbfObjectiveFunction.getCost(candOut);
                    removableCandidates.add(new RemovalCandidate(candOut, cost));
                }
            }

            // If no sets can be removed, construction is done.
            if (removableCandidates.isEmpty()) {
                break;
            }

            // 3. Build the Restricted Candidate List (RCL) for removals.
            Collections.sort(removableCandidates);
            double minCost = removableCandidates.get(0).cost;
            double maxCost = removableCandidates.get(removableCandidates.size() - 1).cost;
            double threshold = minCost + alpha * (maxCost - minCost);
            
            List<RemovalCandidate> rcl = new ArrayList<>();
            for (RemovalCandidate cand : removableCandidates) {
                if (cand.cost <= threshold) {
                    rcl.add(cand);
                } else {
                    break;
                }
            }

            // 4. Randomly select a candidate from the RCL and remove it.
            RemovalCandidate chosenToRemove = rcl.get(random.nextInt(rcl.size()));
            currentSolution.remove(chosenToRemove.id);
        }

        // 5. Evaluate and return the final constructed solution.
        ObjFunction.evaluate(currentSolution);
        return currentSolution;
    }
    
    
    private Solution<Integer> constructRandomPlusGreedy(int p) {
        Solution<Integer> solution = createEmptySol();
        ArrayList<Integer> CL = makeCL();

        // --- Fase 1: passos aleatórios ---
        for (int step = 0; step < p && !CL.isEmpty(); step++) {
            // escolhe aleatoriamente da CL
            Integer chosen = CL.get(random.nextInt(CL.size()));
            solution.add(chosen);
            CL.remove(chosen);
            ObjFunction.evaluate(solution);
        }

        // --- Fase 2: completa de forma puramente gulosa ---
        while (!CL.isEmpty()) {
            Integer bestCand = null;
            double bestDelta = Double.POSITIVE_INFINITY;

            for (Integer cand : CL) {
                double deltaCost = ObjFunction.evaluateInsertionCost(cand, solution);
                if (deltaCost < bestDelta) {
                    bestDelta = deltaCost;
                    bestCand = cand;
                }
            }

            if (bestCand == null) break;
            solution.add(bestCand);
            CL.remove(bestCand);
            ObjFunction.evaluate(solution);
        }

        return solution;
    }
    
    
    private Solution<Integer> constructSampledGreedy(int sampleSize) {
	    Solution<Integer> solution = createEmptySol();
	    ArrayList<Integer> CL = makeCL();
	
	    while (!CL.isEmpty()) {
	        // 1. Sorteia uma amostra da CL
	        ArrayList<Integer> sample = new ArrayList<>(CL);
	        Collections.shuffle(sample, random);
	
	        if (sample.size() > sampleSize) {
	            sample = new ArrayList<>(sample.subList(0, sampleSize));
	        }
	
	        // 2. Encontra min e max custos dentro da amostra
	        double maxCost = Double.NEGATIVE_INFINITY;
	        double minCost = Double.POSITIVE_INFINITY;
	
	        for (Integer cand : sample) {
	            double deltaCost = ObjFunction.evaluateInsertionCost(cand, solution);
	            if (deltaCost < minCost) minCost = deltaCost;
	            if (deltaCost > maxCost) maxCost = deltaCost;
	        }
	
	        // 3. Constrói RCL só com base na amostra
	        ArrayList<Integer> RCL = new ArrayList<>();
	        for (Integer cand : sample) {
	            double deltaCost = ObjFunction.evaluateInsertionCost(cand, solution);
	            if (deltaCost <= minCost + alpha * (maxCost - minCost)) {
	                RCL.add(cand);
	            }
	        }
	
	        if (RCL.isEmpty()) break;
	
	        // 4. Escolhe aleatoriamente um candidato da RCL
	        Integer chosen = RCL.get(random.nextInt(RCL.size()));
	
	        // 5. Atualiza solução
	        solution.add(chosen);
	        CL.remove(chosen);
	        ObjFunction.evaluate(solution);
	    }
	
	    return solution;
	}

    /**
     * The local search phase of the GRASP iteration. It receives a feasible
     * solution and a searchStrategy.
     *
     * @param solution The solution to be improved.
     * @param strategy The local search strategy to be used (FIRST_IMPROVING or BEST_IMPROVING).
     * @return The improved solution (a local optimum).
     */
    public Solution<Integer> localSearch(Solution<Integer> solution, SearchStrategy strategy) {
        if (strategy == SearchStrategy.FIRST_IMPROVING) {
            return firstImprovingLocalSearch(solution);
        } else {
            return bestImprovingLocalSearch(solution);
        }
    }

    /**
     * Best-improving implementation: evaluates all possible exchanges and applies the best one
     * (if any) that improves the solution. Repeats until no further improvement is possible.
     */
    private Solution<Integer> bestImprovingLocalSearch(Solution<Integer> solution) {
        Double maxDeltaCost;
        Integer bestCandIn = null, bestCandOut = null;

        do {
            maxDeltaCost = 0.0;

            ArrayList<Integer> CL = new ArrayList<>();
            for (int i = 0; i < scqbfObjectiveFunction.getDomainSize(); i++) {
                if (!solution.contains(i)) {
                    CL.add(i);
                }
            }

            for (Integer candOut : new ArrayList<>(solution)) {
                for (Integer candIn : CL) {
                    
                    Solution<Integer> tempSol = new Solution<>(solution);
                    tempSol.remove(candOut);
                    tempSol.add(candIn);

                    if (scqbfObjectiveFunction.isFeasible(tempSol)) {
                        double deltaCost = scqbfObjectiveFunction.evaluateExchangeCost(candIn, candOut, solution);

                        if (deltaCost > maxDeltaCost) {
                            maxDeltaCost = deltaCost;
                            bestCandIn = candIn;
                            bestCandOut = candOut;
                        }
                    }
                }
            }

            if (maxDeltaCost > 1e-9) {
                solution.remove(bestCandOut);
                solution.add(bestCandIn);
                scqbfObjectiveFunction.evaluate(solution);
            }

        } while (maxDeltaCost > 1e-9);

        return solution;
    }
    
    /**
     * First-improving implementation: applies the first improving exchange found.
     * As soon as a delta < 0 is found, it applies it and restarts the search.
     */
    private Solution<Integer> firstImprovingLocalSearch(Solution<Integer> solution) {
        boolean improvementFound;

        do {
            improvementFound = false;

            ArrayList<Integer> CL = new ArrayList<>();
            for (int i = 0; i < scqbfObjectiveFunction.getDomainSize(); i++) {
                if (!solution.contains(i)) {
                    CL.add(i);
                }
            }

            searchLoop: 
            for (Integer candOut : new ArrayList<>(solution)) {
                for (Integer candIn : CL) {
                    
                    Solution<Integer> tempSol = new Solution<>(solution);
                    tempSol.remove(candOut);
                    tempSol.add(candIn);

                    if (scqbfObjectiveFunction.isFeasible(tempSol)) {
                        double deltaCost = scqbfObjectiveFunction.evaluateExchangeCost(candIn, candOut, solution);

                        if (deltaCost > 1e-9) {
                            solution.remove(candOut);
                            solution.add(candIn);
                            scqbfObjectiveFunction.evaluate(solution);
                            
                            improvementFound = true;
                            break searchLoop; 
                        }
                    }
                }
            }

        } while (improvementFound);

        return solution;
    }

    /**
     * Creates a full solution where all possible sets are included.
     */
    protected Solution<Integer> createFullSolution() {
        Solution<Integer> sol = new Solution<>();
        for (int i = 0; i < ObjFunction.getDomainSize(); i++) {
            sol.add(i);
        }
        return sol;
    }
    
    @Override 
    public Solution<Integer> createEmptySol() {
        Solution<Integer> sol = new Solution<>();
        sol.cost = 0.0;
        return sol;
    }
    
    @Override public ArrayList<Integer> makeCL() {
        ArrayList<Integer> cl = new ArrayList<>();
        for (int i = 0; i < ObjFunction.getDomainSize(); i++) {
            cl.add(i);
        }
        return cl;
    }
    
    
    @Override public ArrayList<Integer> makeRCL() { return null; }
    @Override public void updateCL() { }
    @Override public Solution<Integer> localSearch() { return null; }
	

    /**
     * A main method for testing the GRASP_SCQBF solver.
     */
    public static void main(String[] args) throws IOException {
        String instancePath = "C:/Users/lilia/OneDrive/Estudo/Otimização Combinatório - projeto/T2/framework bruno/GRASP-Framework/GRASP-Framework/GRASP-Framework/instances/instancias_novas/instancia_01.txt";

        // Configurações de alpha
        double alpha1 = 0.15;
        double alpha2 = 0.50;
        int iterations = 1000;

        // 1. PADRÃO
        runExperiment("PADRÃO", alpha1, ConstructionType.SUBTRACTIVE, SearchStrategy.FIRST_IMPROVING, iterations, instancePath);

        // 2. PADRÃO + ALPHA
        runExperiment("PADRÃO+ALPHA", alpha2, ConstructionType.SUBTRACTIVE, SearchStrategy.FIRST_IMPROVING, iterations, instancePath);

        // 3. PADRÃO + BEST
        runExperiment("PADRÃO+BEST", alpha1, ConstructionType.SUBTRACTIVE, SearchStrategy.BEST_IMPROVING, iterations, instancePath);

        // 4. PADRÃO + HC1
        runExperiment("PADRÃO+HC1", alpha1, ConstructionType.RANDOM_GREEDY, SearchStrategy.FIRST_IMPROVING, iterations, instancePath);

        // 5. PADRÃO + HC2
        runExperiment("PADRÃO+HC2", alpha1, ConstructionType.SAMPLED_GREEDY, SearchStrategy.FIRST_IMPROVING, iterations, instancePath);
    }

    // Função auxiliar para reduzir repetição
    private static void runExperiment(String label, double alpha, ConstructionType construction,
                                      SearchStrategy strategy, int iterations, String instancePath) throws IOException {
        long start = System.currentTimeMillis();

        GRASP_SCQBF grasp = new GRASP_SCQBF(alpha, iterations, instancePath, construction, strategy);
        Solution<Integer> sol = grasp.solve();

        long end = System.currentTimeMillis();
        double totalTime = (end - start) / 1000.0;

        System.out.printf("%s (alpha=%.2f, constr=%s, search=%s)%n", 
            label, alpha, construction, strategy);
        System.out.println(" → Solução: " + sol);
        System.out.println(" → Tempo: " + totalTime + " seg\n");
    }

   
    
    
}