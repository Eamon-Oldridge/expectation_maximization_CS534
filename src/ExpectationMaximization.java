import java.util.ArrayList;

public class ExpectationMaximization {

    private ArrayList<ArrayList<Double>> values;

    /**
     * constructor
     */
    public ExpectationMaximization(ArrayList<ArrayList<Double>> values) {
        this.values = values;
    }

    /**
     * starts the expectation maximization algorithm
     * does BIC first if n = 0
     * @param n the number of clusters to generate
     */
    public void start(int n, boolean printLL, boolean printProb) {
        System.out.println("Running expectation maximization...\n");
        long startTime = System.currentTimeMillis();
        if (n == 0) n = getBIC(); //run BIC and keep track of time if n == 0
        long timeElapsed = System.currentTimeMillis() - startTime;
        Result result = runEM(10000 - timeElapsed, n, printLL, printProb);

        //print results
        ArrayList<Gaussian> gaussians = result.getGaussians();
        for (int i  = 0; i < gaussians.size(); i++) {
            System.out.println("Cluster " + (i + 1) + ":");
            Gaussian gaussian = gaussians.get(i);
            ArrayList<Double> mean = gaussian.getMean();
            ArrayList<Double> variance = gaussian.getVariance();
            for (int j = 0; j < mean.size(); j++) {
                System.out.print("Dimension " + (j + 1) + ": ");
                System.out.print("Mean " + mean.get(j) + ", ");
                System.out.println("Variance " + variance.get(j));
            }
            System.out.println();
        }
        System.out.println("Log likelihood: " + result.getLL());
        double k = result.getGaussians().size() * (2 * values.get(0).size() + 1); //number of parameters
        double num = values.size(); //sample size
        double BIC = k * Math.log(num) - 2 * result.getLL();
        System.out.println("BIC: " + BIC);
    }

    /**
     * runs the expectation maximization algorithm
     * @param time how long to run the algorithm for (in ms)
     * @param n number of clusters
     */
    private final double RESTART = 0.01;
    private final int MAXRESTART = 20;
    private Result runEM(long time, int n, boolean printLL, boolean printProb) {
        //randomly initialize gaussians
        ArrayList<Gaussian> gaussians = getInitialGaussians(n);
        ArrayList<Gaussian> initial = new ArrayList<Gaussian>(gaussians);

        //store the best results
        ArrayList<Gaussian> bestGaussians = gaussians;
        double bestLL = -1000000;
        ArrayList<Gaussian> bestInitial = new ArrayList<Gaussian>(initial);

        //stores the probabilities that each point belongs to each gaussian, where
        //each row represents a point
        //each column represents one of the gaussians/clusters
        double[][] probabilities = new double[values.size()][values.get(0).size()];

        //run while fewer than input ms have elapsed
        long startTime = System.currentTimeMillis();
        double LL = 0;
        double previousLL = -1000000;
        int restarts = -1; //first restart doesn't count becuase of intial previousLL value
        while (System.currentTimeMillis() - startTime < time) {
            probabilities = expectation(gaussians); //EXPECTATION!
            previousLL = LL;
            LL = getLogLikelihood(probabilities);
            if (printLL) System.out.println(LL);
            probabilities = normalize(probabilities);
            gaussians = maximization(probabilities); //MAXIMIZATION!

            //if new LL is better than current best one, store it
            if (LL > bestLL) {
                bestGaussians = gaussians;
                bestLL = LL;
                //store the initial cluster centers that produces this result
                bestInitial = new ArrayList<Gaussian>(initial);
            }

            //handle restarting
            if (LL - previousLL < RESTART && restarts < MAXRESTART) {
                gaussians = getInitialGaussians(n);
                initial = new ArrayList<Gaussian>(gaussians);
                restarts++;
            }

            //once we have used up maximum restarts
            //run with the initial cluster centers that produced best results for reainder of time
            if (restarts == MAXRESTART) {
                gaussians = getInitialGaussians(n);
                restarts++;
            }
        }
        if(printProb) {
            for (int i = 0; i < values.size(); i++) {
                for (int j = 0; j < n; j++) {
                    System.out.print(probabilities[i][j] + " ");
                }
                System.out.println();
            }
        }
        //return a Result object that stores gaussians and LL
        return new Result(bestGaussians, bestLL);
    }

    /**
     * runs bayesian information criterion (BIC)
     * to determine optimal number of clusters
     */
    private int getBIC() {
        ArrayList<Double> BICS = new ArrayList<Double>();
        for (int i = 1; i <= 20; i++) {
            Result result = runEM(250, i, false, false);
            double k = result.getGaussians().size() * (2 * values.get(0).size() + 1); //number of parameters
            double n = values.size(); //sample size
            double BIC = k * Math.log(n) - 2 * result.getLL();
            BICS.add(BIC);
        }
        //take the minimum value
        int min = 0;
        double minValue = BICS.get(0);
        for (int i = 0; i < 20; i++) {
            if (BICS.get(i) < minValue) {
                min = i;
                minValue = BICS.get(i);
            }
        }
        min++; //plus one because of zero-indexing
        return min;
    }

     /**
     * generates n random gaussians
     * @param n number of gaussians to generate
     */
    private ArrayList<Gaussian> getInitialGaussians(int n) {
        ArrayList<Gaussian> gaussians = new ArrayList<Gaussian>();
        ArrayList<Integer> random = new ArrayList<Integer>();
        for (int i = 0; i < n; i++) { //generate n unique random numbers
            boolean valid = false;
            while (!valid) { //do this until we find a valid random number
                int r = (int) (Math.random() * values.size());
                if (!random.contains(r)) { //this random number is valid if we have not already used it
                    random.add(r);
                    valid = true;
                    //the value at position r is a cluster center
                    ArrayList<Double> mean = new ArrayList<Double>(values.get(r));

                    //calculate variance for each cluster using entire data set
                    int size = mean.size();
                    ArrayList<Double> variance = new ArrayList<Double>();
                    for (int j = 0; j < size; j++) {
                        double sum = 0;
                        for (int k = 0; k < values.size(); k++) {
                            double value = values.get(k).get(j);
                            sum += Math.pow(mean.get(j) - value, 2);
                        }
                        sum /= values.size();
                        variance.add(sum);
                    }

                    Gaussian g = new Gaussian(mean, variance);
                    gaussians.add(g);
                }
            }
        }
        return gaussians;
    }

    /**
     * given the probabilities of each point belonging to each cluster (not normalized),
     * calculate the log likelihood of the data
     */
    private double getLogLikelihood(double[][] probabilities) {
        double logLikelihood = 0;
        for (int i = 0; i < probabilities.length; i++) {
            double sum = 0;
            for (int j = 0; j < probabilities[0].length; j++) {
                sum += probabilities[i][j];
            }
            sum = Math.log(sum);
            logLikelihood += sum;
        }
        return logLikelihood;
    }

    /**
     * runs the expectation step of the algorithm
     * given the cluster means and variances,
     * calculate the probability of each point belonging to each cluster
     */
    private double[][] expectation(ArrayList<Gaussian> gaussians) {
        double[][] probabilities = new double[values.size()][gaussians.size()];

        for (int i = 0; i < values.size(); i++) { //iterate through points
            ArrayList<Double> point = values.get(i);
            for (int j = 0; j < gaussians.size(); j++) { //repeat for each gaussian
                Gaussian gaussian = gaussians.get(j);
                double probability = 1;
                //need to multiply by value for each dimension of the point
                for (int k = 0; k < point.size(); k++) {
                    double value = point.get(k);
                    double mean = gaussian.getMean().get(k);
                    double variance = gaussian.getVariance().get(k);
                    probability *= expectationFunction(value, mean, variance);
                }
                probabilities[i][j] = probability;
            }
        }
        return probabilities;
    }

    /**
     * calculate the probability that a value belongs to a gaussian with given mean and variance
     */
    private double expectationFunction(double value, double mean, double variance) {
        double n1 = 1/(Math.sqrt(2 * Math.PI * variance));
        double n2 = -(Math.pow((value-mean), 2))/(2*variance);
        double n3 = n1*Math.pow(Math.E, n2);
        return n3;
    }

    /**
     * given a list of groups of probabilities, normalize each row
     */
    private double[][] normalize(double[][] probabilities) {
        for (int i = 0; i < probabilities.length; i++) {
            double sum = 0;
            for (int j = 0; j < probabilities[0].length; j++) {
                sum += probabilities[i][j];
            }
            for (int j = 0; j < probabilities[0].length; j++) {
                probabilities[i][j] /= sum;
            }
        }
        return probabilities;
    }

    /**
     * runs the maximization step of the algorithm
     * given the probabilities of each point belonging to each cluster,
     * recalculate the cluster means and variances
     */
    private ArrayList<Gaussian> maximization(double[][] probabilities) {
        ArrayList<Gaussian> gaussians = new ArrayList<Gaussian>();

        for (int i = 0; i < probabilities[0].length; i++) { //do for each gaussian
            ArrayList<Double> means = new ArrayList<Double>();
            ArrayList<Double> variances = new ArrayList<Double>();
            for (int j = 0; j < values.get(0).size(); j++) { //for each dimension of the point
                double numerator1 = 0;
                double denominator = 0;
                //CALCULATING NEW MEAN
                for (int k = 0; k < probabilities.length; k++) { //for each point
                    ArrayList<Double> point = values.get(k);
                    numerator1 += probabilities[k][i]*point.get(j);
                    denominator += probabilities[k][i];
                }
                double mean = numerator1/denominator;
                means.add(mean);
                double numerator2 = 0;
                //CALCULATING NEW VARIANCE
                for (int k = 0; k < probabilities.length; k++) { //for each point
                    ArrayList<Double> point = values.get(k);
                    numerator2 += probabilities[k][i]*Math.pow(point.get(j)-mean,2);
                }
                double variance = numerator2/denominator;
                variances.add(variance);
            }
            Gaussian g = new Gaussian(means, variances);
            gaussians.add(g);
        }

        return gaussians;
    }

}