import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from scipy.interpolate import interpn
import statsmodels.api as sm

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

        self.mean_scatter_plot=pd.DataFrame()
        self.max_value=float()

    def get_scatter_mean_plot_df(self):
        """
        Calculate the mean transcript values and missing values count for proteins.

        This method calculates the mean transcript values and the count of missing values (MVs) for proteins.
        It uses the `transcriptomics_unstack` and `proteomics_unstack` dataframes to perform the calculations.
        The mean transcript values are obtained by grouping the `transcriptomics_unstack` dataframe by level 1 and taking the mean.
        The count of missing values for each protein is obtained by grouping the `proteomics_unstack` dataframe by level 1 and applying a lambda function that counts the number of NaN values.

        Returns:
        - mean_scatter_plot: A pandas DataFrame containing the mean transcript values and MVs counts for proteins.

        Example usage:
        >>> obj = MyClass()
        >>> obj.get_scatter_mean_plot_df()

        """


        #obtain the means of transcript values
        mean_transcript=self.transcriptomics_unstack.groupby(level=1).mean()
        MVs_proteins = self.proteomics_unstack.groupby(level=1).apply(lambda x: x.isna().sum())

        self.mean_scatter_plot=pd.DataFrame({'mean_transcript_value':mean_transcript, 'MVs_counts':MVs_proteins})
        self.max_value=max(self.mean_scatter_plot.max())

    def get_scatter_mean_plot_graph(self, with_lowess, plot_points):
        
        sns.regplot(x='mean_transcript_value', y='MVs_counts', data=self.mean_scatter_plot, lowess=with_lowess, truncate=True, scatter_kws={'color': 'white', 'edgecolor': 'black', 'linewidth': 0.2}, ci=None, scatter=plot_points, line_kws={'color': 'black'})

    def get_heatmap_mean_plot(self):
        sns.kdeplot(data=self.mean_scatter_plot , x='mean_transcript_value', y='MVs_counts', fill=True, cmap='viridis')

    def get_hex_bins_graph(self, bin_size=20):
        plt.clf
        sns.jointplot(data=self.mean_scatter_plot,x='mean_transcript_value', y='MVs_counts', kind='hex', cmap='Blues', joint_kws={'gridsize':bin_size})
    
    def get_capped_lowess_scatter_plot(self):
        self.point_coloring_heatmap(False)

        #plot the capped lowess
        x=self.mean_scatter_plot['mean_transcript_value']
        y=self.mean_scatter_plot['MVs_counts']
        lowess = sm.nonparametric.lowess(y, x, frac=0.3)
        y_smooth_capped = np.clip(lowess[:,1], 0, self.max_value)        
        plt.plot(lowess[:,0], y_smooth_capped, color='black')


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
        # histogram_dataset=self.transcriptomics_unstack[MV_mask]

        counts_transcripts, bins, _ = plt.hist(self.transcriptomics_unstack, bins=bin_number)
        
        df=pd.concat([self.transcriptomics_unstack, self.proteomics_unstack], axis=1)
        df=df.drop(df.index[~df.isnull().any(axis=1)])

        counts_MVs, _, patches = plt.hist(df.iloc[:,0], bins)
        percentages= counts_MVs/counts_transcripts
        plt.clf()

        for i in range(len(patches)):
            plt.text(patches[i].get_x()+patches[i].get_width()/2, (patches[i].get_height()+5)/2, f'{percentages[i]:.2f}', ha='center', fontsize=6, rotation=90 )


        #add 0 to the first bin
        bins_2=[0, *bins]
        axis.set_xticks(bins)
        axis.set_xticklabels([f'{bins_2[i]:.2f}-{bins_2[i+1]:.2f}' for i in range(len(bins_2)-1)])
        plt.xlabel("Transcript Values")
        plt.ylabel("Nº of MVs/Proteins")
        plt.size=(10,10)

        #plot the histogram with the bars for each bin side by side 
        plt.bar(bins[:-1], counts_transcripts, width=(bins[1]-bins[0])/2, alpha=0.8, label='Number of Proteins per interval', align='center')
        plt.bar(bins[:-1]+((bins[1]-bins[0])/2), counts_MVs, width=(bins[1]-bins[0])/2, alpha=0.8, label='Number of MVs per interval', align='center')
        plt.legend()
        plt.savefig('histogram_MVs.png')
        plt.clf()

        #calculate the percentage of proteins per bin
        percentages_proteins=counts_transcripts/sum(counts_transcripts)
        percentages_MVs=counts_MVs/sum(counts_MVs)

        #make a table with the protein counts, MVs counts and percentages of MVs per bin with only 2 decimal places
        table=pd.DataFrame({'Proteins':counts_transcripts, 'MVs':counts_MVs, 'Percentage in bin':percentages, 'Percentage of total proteins':percentages_proteins, 'Percentage of total MVs':percentages_MVs})
        table.index=[f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)]
        table.to_csv('histogram_table.csv')
        table=table.round(3)
        #make a table where everything fits and save it as a png
        plt.figure(figsize=(10,10))
        plt.table(cellText=table.values, colLabels=table.columns, rowLabels=table.index, loc='center', cellLoc='center')
        plt.axis('off')
        plt.savefig('histogram_table.png')
        plt.clf()








        


if __name__=='__main__':
    #open a new folder to save the results
    import os
    if not os.path.exists('results/results_dataframe_distributions'):
        os.makedirs('results/results_dataframe_distributions')

    os.chdir('results/results_dataframe_distributions')

    df=MVS_dataframe('/data/benchmarks/clines/transcriptomics.csv','/data/benchmarks/clines/proteomics.csv')
    df.get_scatter_mean_plot_df()

    save_fig, axis = plt.subplots()
    #axis.scatter(df.mean_scatter_plot['mean_transcript_value'], df.mean_scatter_plot['MVs_counts'])
    axis.set_xlabel('Transcript Mean')
    axis.set_ylabel('Number of MVs')
    plt.xlim(df.mean_scatter_plot["mean_transcript_value"].min(), df.mean_scatter_plot["mean_transcript_value"].max())
    plt.ylim(0, df.mean_scatter_plot['MVs_counts'].max())

    df.get_scatter_mean_plot_graph(True, True)
    plt.savefig("plot_by_mean.png")
    plt.clf()
    df.get_heatmap_mean_plot()
    plt.savefig("plot_by_mean_heatmap.png")
    plt.clf()

    save_fig, axis = plt.subplots()
    df.get_capped_lowess_scatter_plot()
    plt.ylim(-1, 1+df.mean_scatter_plot['MVs_counts'].max())
    axis.set_xlabel('Transcriptomic Mean Value')
    axis.set_ylabel('Number of MVs')
    plt.savefig('plot_by_mean_lowess_capped.png')
    plt.clf()

    df.point_coloring_heatmap(True)
    plt.savefig('plot_by_mean_point_color.png')
    plt.clf()
    
    df.get_hex_bins_graph()
    plt.clf()

    df.get_histogram(20)



# plt.xlabel("Transcrip Values")
# plt.ylabel("Nº of MVs")
# plt.title("Histogram of MVs in intervals of Transcript Values")





