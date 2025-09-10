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
            // Sort in descending order of cost
            return other.cost.compareTo(this.cost);
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
    public GRASP_SCQBF(Double alpha, Integer iterations, String filename) throws IOException {
        super(new SCQBF(filename), alpha, iterations);
        this.scqbfObjectiveFunction = (SCQBF) this.ObjFunction;
    }

    /**
     * The main solver method. It runs the GRASP iterations, performing the
     * construction and local search phases, and returns the best solution found.
     */
    @Override
    public Solution<Integer> solve() {
        Solution<Integer> bestSol = null;

        for (int i = 0; i < iterations; i++) {
            // Phase 1: Construction
            Solution<Integer> currentSol = runSingleSubtractiveIteration();
            
            // Phase 2: Local Search
            currentSol = localSearch(currentSol);

            if (bestSol == null || currentSol.cost < bestSol.cost) {
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
            double maxCost = removableCandidates.get(0).cost;
            double minCost = removableCandidates.get(removableCandidates.size() - 1).cost;
            double threshold = maxCost - alpha * (maxCost - minCost);
            
            List<RemovalCandidate> rcl = new ArrayList<>();
            for (RemovalCandidate cand : removableCandidates) {
                if (cand.cost >= threshold) {
                    rcl.add(cand);
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

    /**
     * The local search phase of the GRASP iteration. It receives a feasible
     * solution and tries to improve it by exploring its 1-1 exchange neighborhood.
     *
     * @param solution The solution to be improved.
     * @return The improved solution (a local optimum).
     */
    public Solution<Integer> localSearch(Solution<Integer> solution) {
        Double minDeltaCost;
        Integer bestCandIn = null, bestCandOut = null;

        do {
            minDeltaCost = 0.0;

            // Create a list of candidates NOT in the solution (CL)
            ArrayList<Integer> CL = new ArrayList<>();
            for (int i = 0; i < scqbfObjectiveFunction.getDomainSize(); i++) {
                if (!solution.contains(i)) {
                    CL.add(i);
                }
            }

            // Evaluate all possible 1-1 exchanges (swaps)
            // Iterate on a copy of the solution to avoid ConcurrentModificationException
            for (Integer candOut : new ArrayList<>(solution)) {
                for (Integer candIn : CL) {
                    
                    // Create a temporary solution to check feasibility of the swap
                    Solution<Integer> tempSol = new Solution<>(solution);
                    tempSol.remove(candOut);
                    tempSol.add(candIn);

                    if (scqbfObjectiveFunction.isFeasible(tempSol)) {
                        // If the swap is feasible, calculate its impact on the cost
                        double deltaCost = scqbfObjectiveFunction.evaluateExchangeCost(candIn, candOut, solution);

                        if (deltaCost < minDeltaCost) {
                            minDeltaCost = deltaCost;
                            bestCandIn = candIn;
                            bestCandOut = candOut;
                        }
                    }
                }
            }

            // If a cost-improving move was found, apply it to the solution
            if (minDeltaCost < -1e-9) { // Use a small epsilon for floating point comparison
                solution.remove(bestCandOut);
                solution.add(bestCandIn);
                scqbfObjectiveFunction.evaluate(solution); // Recalculate cost
            }

        } while (minDeltaCost < -1e-9);

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
    
    private Solution<Integer> constructRandomPlusGreedy(int k) {
    	Solution<Integer> solution = createEmptySol();
    	ArrayList<Integer> CL = makeCL();
    	while (!CL.isEmpty()) {
    		 double maxCost = Double.NEGATIVE_INFINITY;
    	     double minCost = Double.POSITIVE_INFINITY;
    	     for (Integer cand : CL) {
    	            double deltaCost = ObjFunction.evaluateInsertionCost(cand, solution);
    	            if (deltaCost < minCost) minCost = deltaCost;
    	            if (deltaCost > maxCost) maxCost = deltaCost;
    	        }
    	     ArrayList<Integer> RCL = new ArrayList<>();
    	     for (Integer cand : CL) {
    	        double deltaCost = ObjFunction.evaluateInsertionCost(cand, solution);
    	        if (deltaCost <= minCost + alpha * (maxCost - minCost)) {
    	            RCL.add(cand);
    	        }
             }
    	     if (RCL.isEmpty()) break;
    	     // Random plus greedy step - Sorteia até k candidatos da RCL
    	     ArrayList<Integer> sample = new ArrayList<>();
    	     Collections.shuffle(RCL, random);
    	     for (int i = 0; i < Math.min(k, RCL.size()); i++) {
    	        sample.add(RCL.get(i));
    	     }
    	     // Escolhe o melhor da amostra (greedy)
    	     Integer bestCand = null;
    	     double bestDelta = Double.POSITIVE_INFINITY;
    	     for (Integer cand : sample) {
    	        double deltaCost = ObjFunction.evaluateInsertionCost(cand, solution);
    	          if (deltaCost < bestDelta) {
    	              bestDelta = deltaCost;
    	              bestCand = cand;
    	           }
    	      }
    	     // Adiciona o melhor candidato à solução
    	     if (bestCand != null) {
    	         solution.add(bestCand);
    	         CL.remove(bestCand);
    	         ObjFunction.evaluate(solution);
    	     } else {
    	         break;
    	     }
    	}
    	return solution;
    }

    /**
     * A main method for testing the GRASP_SCQBF solver.
     */
    public static void main(String[] args) throws IOException {
        long startTime = System.currentTimeMillis();
        
        GRASP_SCQBF solver = new GRASP_SCQBF(0.15, 1000, "instances/scqbf/instance_9.txt");
        Solution<Integer> bestSol = solver.solve();
        
        System.out.println("Best Solution Found (Min Cost) = " + bestSol);
        
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Time = " + (double) totalTime / 1000 + " seg");
    }

    @Override public Solution<Integer> createEmptySol() { return null; }
    @Override public ArrayList<Integer> makeCL() { return null; }
    @Override public ArrayList<Integer> makeRCL() { return null; }
    @Override public void updateCL() { }
    @Override public Solution<Integer> localSearch() { return null; }
}