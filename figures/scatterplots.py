import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rc('text',usetex=True)
plt.rc('font', family='serif', size=16)


def scatter(scatterdata):
    def plot_model(scatterdata, MODEL, ax):
        MEASURE = "weak%strong-mdl"
        METRIC = "test_f_score"
        scatterdata = scatterdata.copy()  #
        scatterdata = scatterdata[scatterdata.model.str.contains(MODEL)]

        ax = sns.regplot(
            y=METRIC,  # test_f_score
            x=MEASURE,
            data=scatterdata,
            logistic=True,
            scatter_kws={"s": 25},
            ax=ax,
        )
        ax.set_ylabel(r"Average F-Score")
        ax.set_xlabel("Relative extractibility of target feature (MDL($s$)/MDL($t$))")
        ax.set_title(
            MODEL.replace("lstm", "GloVe")
            .replace("t5", "T5")
            .replace("gpt2", "GPT-2")
            .replace("bert", "BERT")
            .replace("roBERTa", "RoBERTa")
        )
        #ax.set_xticks([0.01, 1, 2])

    def plot(scatterdata):
        fig = plt.figure()
        # subplots(
        #     1, 1, figsize=(8, 5), constrained_layout=True, sharex=True, sharey=True
        # )
        # ax1 = axes[0]  # plt.subplot(gs[0])
        # ax2 = axes[1]  # plt.subplot(gs[1])
        # ax3 = axes[2]  # plt.subplot(gs[2])
        # ax4 = axes[3]  # plt.subplot(gs[3])
        # ax5 = axes[4]  # plt.subplot(gs[4])
        X = 0.73
        Y = 0.075
        fontsize = 10

        # plot_model(scatterdata, "bert", ax1)
        # ax1.text(
        #     X,
        #     Y,
        #     r"$\rho = .79*$",
        #     fontsize=fontsize,
        #     horizontalalignment="center",
        #     verticalalignment="center",
        #     bbox={"facecolor": "silver", "alpha": 0.45, "pad": 2},
        #     transform=ax1.transAxes,
        # )
        #ax1.set_ylabel(r"Average F-Score")

        # plot_model(scatterdata, "roberta", ax2)
        # ax2.text(
        #     X,
        #     Y,
        #     r"$\rho = .83*$",
        #     fontsize=fontsize,
        #     horizontalalignment="center",
        #     verticalalignment="center",
        #     bbox={"facecolor": "silver", "alpha": 0.45, "pad": 2},
        #     transform=ax2.transAxes,
        # )

        # plot_model(scatterdata, "t5", ax3)
        # ax3.text(
        #     X,
        #     Y,
        #     r"$\rho = .57*$",
        #     fontsize=fontsize,
        #     horizontalalignment="center",
        #     verticalalignment="center",
        #     bbox={"facecolor": "silver", "alpha": 0.45, "pad": 2},
        #     transform=ax3.transAxes,
        # )
        # plt.gcf().text(
        #     0.5,
        #     -0.05,
        #     "Relative extractibility of target feature (MDL($s$)/MDL($t$))",
        #     horizontalalignment="center",
        #     verticalalignment="bottom",
        #     fontsize=12,
        # )
        #plt.ylabel(r"Average F-Score")

        plot_model(scatterdata, "gpt2", plt.gca())
        # ax1.text(
        #     X,
        #     Y,
        #     fontsize=fontsize,
        #     horizontalalignment="center",
        #     verticalalignment="center",
        #     bbox={"facecolor": "silver", "alpha": 0.45, "pad": 2},
        #     transform=ax1.transAxes,
        # )

        # plot_model(scatterdata, "lstm", ax5)
        # ax5.text(
        #     X,
        #     Y,
        #     r"$\rho = .14 $",
        #     fontsize=fontsize,
        #     horizontalalignment="center",
        #     verticalalignment="center",
        #     bbox={"facecolor": "silver", "alpha": 0.45, "pad": 2},
        #     transform=ax5.transAxes,
        # )
        plt.tight_layout()
        fig.savefig(
            f"figures/scatterplot.pdf", transparent=True, bbox_inches="tight"
        )  # .png", dpi=600)
        plt.close()

    plot(scatterdata)


def main():
    # The scatterdata are controlled.
    scatterdata = pd.read_table("files/scatterdata.tsv")
    scatter(scatterdata)
