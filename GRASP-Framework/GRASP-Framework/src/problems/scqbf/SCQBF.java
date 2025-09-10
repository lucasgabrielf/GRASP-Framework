package problems.scqbf;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import problems.Evaluator;
import solutions.Solution;

/**
 * The Set Cover Quadratic Binary Function (SC-QBF) problem.
 * This class represents a constrained optimization problem where the goal is to
 * find a minimum cost subset of available sets that covers all elements of a universe.
 * The cost function is a Quadratic Binary Function (f(x) = x'Ax), where x is a binary
 * vector representing the chosen sets.
 *
 * @author adapted from ccavellucci, fusberti
 */
public class SCQBF implements Evaluator<Integer> {

    /**
     * The number of elements in the universe to be covered.
     */
    public final Integer numUniverseElements;

    /**
     * The number of available subsets to choose from. This is the domain size.
     */
    public final Integer numSubsets;

    /**
     * An incidence matrix representing the sets. coverage[i][j] is true if
     * universe element 'i' is covered by subset 'j'.
     */
    protected boolean[][] coverage;

    /**
     * The matrix A of coefficients for the QBF cost function f(x) = x'.A.x.
     */
    public Double[][] A;

    /**
     * A binary vector representing the current solution (x). variables[i] = 1.0
     * if subset 'i' is in the solution, 0.0 otherwise.
     */
    public Double[] variables;

    /**
     * Constructor for the SCQBF class. It reads the problem parameters from a
     * specified file.
     *
     * @param filename 
     * @throws IOException 
     */
    public SCQBF(String filename) throws IOException {
        Integer size = readInput(filename);
        this.numUniverseElements = size;
        this.numSubsets = size;
        this.variables = allocateVariables();
    }
    
    /**
     * {@inheritDoc}
     * For SC-QBF, this is the number of available subsets.
     */
    @Override
    public Integer getDomainSize() {
        return this.numSubsets;
    }

    /**
     * Responsible for parsing the problem instance file.
     *
     * @param filename 
     * @return 
     * @throws IOException
     */
    protected Integer readInput(String filename) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            // Part 1: Read the size of the problem (n).
            int size = Integer.parseInt(reader.readLine().trim());

            // Initialize data structures.
            this.A = new Double[size][size];
            this.coverage = new boolean[size][size];

            // Part 2: Read the n linear costs (diagonal of matrix A).
            String[] linearCosts = reader.readLine().trim().split("\\s+");
            for (int i = 0; i < size; i++) {
                A[i][i] = Double.parseDouble(linearCosts[i]);
            }

            // Part 3: Read the n subset definitions.
            for (int i = 0; i < size; i++) {
                String[] coveredElements = reader.readLine().trim().split("\\s+");
                for (String elementStr : coveredElements) {
                    if (!elementStr.isEmpty()) {
                        // Convert from 1-based in file to 0-based in array.
                        int elementIndex = Integer.parseInt(elementStr) - 1;
                        if (elementIndex >= 0 && elementIndex < size) {
                            coverage[elementIndex][i] = true;
                        }
                    }
                }
            }

            // Part 4: Read the n-1 lines of the upper triangular quadratic cost matrix.
            for (int i = 0; i < size - 1; i++) {
                String line = reader.readLine();
                if (line == null) {
                    throw new IOException("Arquivo terminou antes de ler toda a matriz quadrática (i=" + i + ")");
                }
                String[] interactionCosts = line.trim().split("\\s+");

                int expected = size - (i + 1);
                if (interactionCosts.length != expected) {
                    throw new IOException(
                        "Erro na linha " + (i+1) + " da matriz quadrática: esperado " +
                        expected + " valores, mas veio " + interactionCosts.length
                    );
                }

                for (int j = 0; j < expected; j++) {
                    double cost = Double.parseDouble(interactionCosts[j]);
                    int col = i + 1 + j;
                    A[i][col] = cost;
                    A[col][i] = cost;
                }
            }

            
            return size;
        }
    }

    /**
     * Evaluates a given solution.
     * First, it checks if the solution constitutes a valid set cover.
     * If it is not a valid cover, the cost is considered infinite.
     * If it is a valid cover, the cost is calculated using the QBF function (x'Ax).
     *
     * @param sol 
     * @return 
     */
    @Override
    public Double evaluate(Solution<Integer> sol) {
        if (!isFeasible(sol)) {
            return sol.cost = Double.POSITIVE_INFINITY;
        }
        setVariables(sol);
        return sol.cost = evaluateQBF();
    }
    
    public boolean isFeasible(Solution<Integer> sol) {
        Set<Integer> coveredElements = new HashSet<>();
        for (Integer subsetIndex : sol) {
            for (int elementIndex = 0; elementIndex < numUniverseElements; elementIndex++) {
                if (coverage[elementIndex][subsetIndex]) {
                    coveredElements.add(elementIndex);
                }
            }
        }
        return coveredElements.size() == numUniverseElements;
    }

    public boolean isFeasibleAfterRemoval(Integer candToRemove, Solution<Integer> sol) {
        int[] coverageCount = new int[numUniverseElements];
        for (Integer subsetInSol : sol) {
            for (int element = 0; element < numUniverseElements; element++) {
                if (coverage[element][subsetInSol]) {
                    coverageCount[element]++;
                }
            }
        }

        for (int element = 0; element < numUniverseElements; element++) {
            if (coverage[element][candToRemove] && coverageCount[element] == 1) {
                return false;
            }
        }
        
        return true;
    }
    
    public Double getCost(Integer subsetIndex) {
        if (subsetIndex < 0 || subsetIndex >= numSubsets) {
            return Double.POSITIVE_INFINITY;
        }
        return A[subsetIndex][subsetIndex];
    }
    
    public void setVariables(Solution<Integer> sol) {
        resetVariables();
        if (!sol.isEmpty()) {
            for (Integer elem : sol) {
                variables[elem] = 1.0;
            }
        }
    }

    public Double evaluateQBF() {
        double sum = 0.0;
        for (int i = 0; i < numSubsets; i++) {
            if (variables[i] != null && variables[i] == 1.0) {
                for (int j = 0; j < numSubsets; j++) {
                    if (variables[j] != null && variables[j] == 1.0) {
                        sum += A[i][j];
                    }
                }
            }
        }
        return sum;
    }

    @Override
    public Double evaluateInsertionCost(Integer elem, Solution<Integer> sol) {
        setVariables(sol);
        return evaluateInsertionQBF(elem);
    }

    public Double evaluateInsertionQBF(int i) {
        if (variables[i] == 1) return 0.0;
        return evaluateContributionQBF(i);
    }
    
    @Override
    public Double evaluateRemovalCost(Integer elem, Solution<Integer> sol) {
        setVariables(sol);
        return evaluateRemovalQBF(elem);
    }

    public Double evaluateRemovalQBF(int i) {
        if (variables[i] == 0) return 0.0;
        return -evaluateContributionQBF(i);
    }

    @Override
    public Double evaluateExchangeCost(Integer elemIn, Integer elemOut, Solution<Integer> sol) {
        setVariables(sol);
        return evaluateExchangeQBF(elemIn, elemOut);
    }
    
    public Double evaluateExchangeQBF(int in, int out) {
        if (in == out) return 0.0;
        if (variables[in] == 1) return evaluateRemovalQBF(out);
        if (variables[out] == 0) return evaluateInsertionQBF(in);
        
        double sum = 0.0;
        sum += evaluateContributionQBF(in);
        sum -= evaluateContributionQBF(out);
        sum -= (A[in][out] + A[out][in]);
        return sum;
    }

    private Double evaluateContributionQBF(int i) {
        double sum = 0.0;
        for (int j = 0; j < numSubsets; j++) {
            if (i != j && variables[j] == 1.0) {
                sum += (A[i][j] + A[j][i]);
            }
        }
        sum += A[i][i];
        return sum;
    }
    
    protected Double[] allocateVariables() {
        return new Double[numSubsets];
    }

    public void resetVariables() {
        Arrays.fill(variables, 0.0);
    }
    
    public static void main(String[] args) throws IOException {
        SCQBF scqbf = new SCQBF("instances/scqbf/instance_9.txt");

        System.out.println("Successfully loaded instance file.");
        System.out.println("Problem Size (n x m): " + scqbf.numSubsets + " x " + scqbf.numUniverseElements);
        
        System.out.println("Linear cost of Subset 0: " + scqbf.getCost(0));
        System.out.println("Linear cost of Subset 24: " + scqbf.getCost(24));
        
        System.out.println("Interaction cost between Subset 0 and Subset 1: " + scqbf.A[0][1]);
    }
}