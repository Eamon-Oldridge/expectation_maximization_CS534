import java.util.ArrayList;

public class Result {

    private ArrayList<Gaussian> gaussians;
    private double LL;

    public Result(ArrayList<Gaussian> gaussians, double LL) {
        this.gaussians = gaussians;
        this.LL = LL;
    }

    public ArrayList<Gaussian> getGaussians() {
        return gaussians;
    }

    public double getLL() {
        return LL;
    }

}
