import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from scipy.interpolate import interpn

class MVS_dataframe():

    def __init__(self, transcriptomics_path, proteomics_path):
        self.transcriptomics= pd.read_csv(transcriptomics_path,index_col=0)
        self.proteomics= pd.read_csv(proteomics_path, index_col=0)

        #Obtain the intersection between both dfs
        self.genes= list(set(self.transcriptomics.index).intersection(set(self.proteomics.index)))
        self.samples= list(set(self.transcriptomics.columns).intersection(set(self.proteomics.columns)))

        self.proteomics=self.proteomics.reindex(index=self.genes, columns=self.samples)
        self.transcriptomics=self.transcriptomics.reindex(index=self.genes, columns=self.samples)

        #transform to long form
        self.proteomics_unstack=self.proteomics.unstack()
        self.transcriptomics_unstack=self.transcriptomics.unstack()

    def get_scatter_mean_plot_df(self):

        #obtain the means of transcript values
        mean_transcript=self.transcriptomics_unstack.groupby(level=1).mean()
        MVs_proteins = self.proteomics_unstack.groupby(level=1).apply(lambda x: x.isna().sum())

        self.mean_scatter_plot=pd.DataFrame({'mean_transcript_value':mean_transcript, 'MVs_counts':MVs_proteins})

    def get_scatter_mean_plot_graph(self, with_lowess, plot_points):
        sns.regplot(x='mean_transcript_value', y='MVs_counts', data=self.mean_scatter_plot, lowess=with_lowess, scatter_kws={'color': 'white', 'edgecolor': 'black', 'linewidth': 0.2}, ci=None, scatter=plot_points, line_kws={'color': 'black'})

    def get_heatmap_mean_plot(self):
        sns.kdeplot(data=self.mean_scatter_plot , x='mean_transcript_value', y='MVs_counts', fill=True, cmap='viridis')

    def get_hex_bins_graph(self, bin_size=20):
        sns.jointplot(data=plot,x='mean_transcript_value', y='MVs_counts', kind='hex', cmap='Blues', joint_kws={'gridsize':20} )

    def point_coloring_heatmap(self, with_lowess):
        if ~self.mean_scatter_plot.empty:
            color=self.density_interpolate(self.mean_scatter_plot['mean_transcript_value'], self.mean_scatter_plot['MVs_counts'])
            
            ax = plt.gca()

            ax.scatter(
                self.mean_scatter_plot["mean_transcript_value"],
                self.mean_scatter_plot["MVs_counts"],
                c=color,
                marker="o",
                edgecolor="none",
                s=5,
                alpha=0.8,
                cmap="Spectral_r",
            )
        else:
            self.get_scatter_mean_plot_df()
            color=self.density_interpolate(self.mean_scatter_plot['mean_transcript_value'], self.mean_scatter_plot['MVs_counts'])           
            ax = plt.gca()

            ax.scatter(
                self.mean_scatter_plot["mean_transcript_value"],
                self.mean_scatter_plot["MVs_counts"],
                c=color,
                marker="o",
                edgecolor="none",
                s=5,
                alpha=0.8,
                cmap="Spectral_r",
            )
        if with_lowess:
            self.get_scatter_mean_plot_graph(True, False)
    
    def density_interpolate(self, xx, yy):
        data, x_e, y_e = np.histogram2d(xx, yy, bins=20)

        zz = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([xx, yy]).T,
            method="splinef2d",
            bounds_error=False,
        )

        return zz
    
    def get_histogram(self, bin_number):
        MV_mask=self.proteomics_unstack.isna()
        histogram_dataset=self.transcriptomics_unstack[MV_mask]

        counts_transcripts, bins, _ = plt.hist(self.transcriptomics_unstack, bins=bin_number)
        
        df=pd.concat([self.transcriptomics_unstack, self.proteomics_unstack], axis=1)
        df=df.drop(df.index[~df.isnull().any(axis=1)])

        counts_MVs, _, _ = plt.hist(df.iloc[:,0], bins)

        percentages= counts_MVs/counts_transcripts
        print(percentages)

if __name__=='__main__':
    df=MVS_dataframe('/data/benchmarks/clines/transcriptomics.csv','/data/benchmarks/clines/proteomics.csv')
    df.get_scatter_mean_plot_df()
    df.get_scatter_mean_plot_graph(True, True)

    save_fig, axis = plt.subplots()
    #axis.scatter(df.mean_scatter_plot['mean_transcript_value'], df.mean_scatter_plot['MVs_counts'])
    axis.set_xlabel('Transcript Mean')
    axis.set_ylabel('Number of MVs')

    plt.savefig("results/plot_by_mean.png")
    plt.clf()
    df.get_heatmap_mean_plot()
    plt.savefig("results/plot_by_mean_heatmap.png")
    plt.clf()

    df.point_coloring_heatmap(True)
    plt.savefig('results/plot_by_mean_point_color.png')
    plt.clf()

    df.get_histogram(20)
    plt.savefig('results/histogram_20_intervals_MVs.png')



# plt.xlabel("Transcrip Values")
# plt.ylabel("Nº of MVs")
# plt.title("Histogram of MVs in intervals of Transcript Values")





