"""
dealing with multinest out put
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import subplots
from scipy import signal
from scipy.ndimage import gaussian_filter


class CrediblePlot:
    """
    class for plotting results from multinest
    """
    def __init__(self, file):
        """
        read .txt file
        :param file: path to .txt file or np.ndarray
        """
        if isinstance(file, str):
            self.ftxt = np.genfromtxt(file)
        elif isinstance(file, np.ndarray):
            self.ftxt = file.copy()

        # if abs(sum(self.ftxt[:, 0])-1) > 1e-3:
        #     raise Exception("Invalid file!")

    def credible_1d(self, idx: int, credible_level=(0.6827, 0.9545), nbins=80, ax=None,
                    give_max=False, label='', smooth=False, countour=True, give_edge=False,
                    color='b', ls='-', flip_axes=False, lwidth=2):
        """
        plot binned parameter v.s. its probability on the bin
        :param idx: which parameter should be plotted? index starts from 0
        :param credible_level: color different credible level, default is 1sigma and 2sigma
        :param nbins: number of bins
        :param ax: axes to be plot on, if not none
        :param give_max: print maximum posterior distribution position, default False
        :param label: label for the plot, string
        :param smooth: whether to show smoothed probability density plot, default Fase, if True, will turn countour to False
        :param countour: draw countour plot with confidence region
        :param give_edge: print edge of the contour
        :return: figure and axes object for further fine tuning the plot
        """
        contour = True
        if ax is not None:
            fig = None
        else:
            fig, ax = subplots()
        minx = np.amin(self.ftxt[:, idx+2])
        maxx = np.amax(self.ftxt[:, idx+2])
        binw = (maxx - minx) / nbins
        binx = np.linspace(minx + binw/2, maxx - binw/2, nbins)
        biny = np.zeros_like(binx)
        for i in range(self.ftxt.shape[0]):
            pos = int((self.ftxt[i, idx+2] - minx) / binw)
            if pos < nbins:
                biny[pos] += self.ftxt[i, 0]
            else:
                biny[pos-1] += self.ftxt[i, 0]
        cl = np.sort(credible_level)[::-1]
        if smooth:
            countour = False
            by = signal.savgol_filter(biny, 5, 2)
            if flip_axes:
                ax.plot(by / binw, binx, label=label, color=color, ls=ls, linewidth=lwidth) # 6 for grid, 2 for 1d
            else:
                ax.plot(binx, by/binw, label=label, color=color, ls=ls, linewidth=lwidth)
        else:
            ax.bar(binx, biny, label=label, width=binw, alpha=0.5, color=color)
        if give_max:
            print(binx[np.argmax(biny)])
        sorted_idx = np.argsort(biny)[::-1]
        if countour:
            al = np.linspace(0.2, 0.3, len(cl))
            for ic in range(len(cl)):
                s = 0
                cxl = []
                cyl = []
                for i in sorted_idx:
                    s += biny[i]
                    cyl.append(biny[i])
                    cxl.append(binx[i])
                    if s > cl[ic]:
                        break
                ax.bar(cxl, cyl, width=binw, alpha=al[ic], color=color)
                if give_edge:
                    print(cl[ic], '-->', np.sort(cxl))
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop)))
        return fig, ax

    def credible_2d(self, idx: tuple, credible_level=(0.6827, 0.9545), nbins=80, ax=None,
                    center=None, heat=False, xlim=None, ylim=None, mark_best=False, color='b', filter_sigma=None):
        """
        plot the correlation between parameters
        :param idx: the index of the two parameters to be ploted
        :param credible_level: choose which credible levels to plot
        :param nbins: number of bins
        :param ax: axes to be plot on, if not none
        :param center: mark center point
        :param heat: enable heat plot or not, default False
        :param xlim: plot range for x axis
        :param ylim: plot range for y axis
        :return: figure and axes object for further fine tuning the plot, if heat, return im
        """
        if ax is not None:
            fig = None
        else:
            fig, ax = subplots()
        minx = np.amin(self.ftxt[:, idx[0]+2])
        miny = np.amin(self.ftxt[:, idx[1]+2])
        maxx = np.amax(self.ftxt[:, idx[0]+2])
        maxy = np.amax(self.ftxt[:, idx[1]+2])
        binxw = (maxx - minx) / nbins
        binyw = (maxy - miny) / nbins
        binx = np.linspace(minx + binxw/2, maxx - binxw/2, nbins)
        biny = np.linspace(miny + binyw/2, maxy - binyw/2, nbins)
        xv, yv = np.meshgrid(binx, biny)
        zv = np.zeros_like(xv)
        # be careful that position in x direction is column, position in y direction is row!
        for i in range(self.ftxt.shape[0]):
            posx = int((self.ftxt[i, idx[0]+2] - minx) / binxw)
            posy = int((self.ftxt[i, idx[1]+2] - miny) / binyw)
            if posx < nbins and posy < nbins:
                zv[posy, posx] += self.ftxt[i, 0]
            elif posy < nbins:
                zv[posy, posx-1] += self.ftxt[i, 0]
            elif posx < nbins:
                zv[posy-1, posx] += self.ftxt[i, 0]
            else:
                zv[posy-1, posx-1] += self.ftxt[i, 0]

        sorted_idx = np.unravel_index(np.argsort(zv, axis=None)[::-1], zv.shape)
        if mark_best:
            print(xv[sorted_idx[0][0], sorted_idx[1][0]], yv[sorted_idx[0][0], sorted_idx[1][0]])
            ax.plot([xv[sorted_idx[0][0], sorted_idx[1][0]]], [yv[sorted_idx[0][0], sorted_idx[1][0]]], '*')
        if heat:
            im = ax.pcolormesh(xv, yv, zv/(binxw*binyw), cmap='rainbow', edgecolors='face')
            if xlim is not None:
                ax.set_xlim(xlim[0], xlim[1])
                ax.set_ylim(ylim[0], ylim[1])
            xleft, xright = ax.get_xlim()
            ybottom, ytop = ax.get_ylim()
            ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)))
            return im
        else:
            cl = np.sort(credible_level)[::-1]
            al = np.linspace(0.2, 0.3, len(cl))
            cll = 0
            if filter_sigma:
                # zv = signal.savgol_filter(zv, 5, 2)
                zv = gaussian_filter(zv, filter_sigma)
            for ic in range(len(cl)):
                cz = np.zeros_like(zv)
                s = 0
                for i in range(sorted_idx[0].shape[0]):
                    s += zv[sorted_idx[0][i], sorted_idx[1][i]]
                    cz[sorted_idx[0][i], sorted_idx[1][i]] = zv[sorted_idx[0][i], sorted_idx[1][i]]
                    if s > cl[ic]:
                        cll = zv[sorted_idx[0][i], sorted_idx[1][i]]
                        break
                ax.contourf(xv, yv, zv, (cll, 1), colors=(color, 'white'), alpha=al[ic])
            ax.axis('scaled')
        if center is not None:
            ax.plot(np.array([center[0]]), np.array([center[1]]), "*")
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)))
        return fig, ax, (xv, yv, zv)

    def credible_grid(self, idx: tuple, test_point: tuple, names=None,
                      credible_level=(0.6827, 0.9545), nbins=80, color="b"):
        """
        n by n grid of plots where diagonal plots are parameters vs probability,
        off diagonal plots are the correlation between any of the two
        :param idx: the indexes of the parameters to be ploted
        :param names: names of the parameters to be placed at axis
        :param credible_level: choose which credible levels to plot
        :param nbins: number of bins
        :return: fig and list of axes
        """
        lth = len(idx)
        fig = plt.figure(figsize=(lth*5, lth*5))
        grid = fig.add_gridspec(lth, lth)
        axes = [[None]*lth]*lth
        for i in range(lth):
            for j in range(i+1):
                axes[i][j] = fig.add_subplot(grid[i, j])
                ax = axes[i][j]
                if i == j:
                    flip = True
                    if i==0:
                        flip = False
                    self.credible_1d(i, credible_level, nbins, ax, color=color, flip_axes=flip)
                    if names is not None:
                        ax.tick_params(labelsize=25)
                        if flip == False:
                            ax.set_ylabel('p', fontsize=35)
                        else:
                            ax.set_ylabel('p', fontsize=35)
                            #ax.xaxis.set_label_position('top')
                            #ax.xaxis.tick_top()
                else:
                    self.credible_2d((j, i), credible_level, nbins, ax, color=color)
                    plt.plot(test_point[j], test_point[i], marker="*", c='r', markersize='20')
                    ax.tick_params(labelsize=25)
                    if names is not None and i==lth-1:
                        ax.set_xlabel(names[j], fontsize=35)
                    if names is not None and j==0:
                        ax.set_ylabel(names[i], fontsize=35)
        fig.tight_layout()
        return fig, axes

    def cumulative_1d(self, idx: int, ax=None, color=None, label=None):
        """
        plot cumulative distribution
        :param idx: index to be plotted
        :param ax: axes to be plot on, if not none
        :param color: color of the line
        :param label: label for the plot, string
        :return: figure and axes object for further fine tuning the plot
        """
        if ax is not None:
            fig = None
        else:
            fig, ax = subplots()
        sorted_idx = np.argsort(self.ftxt[:, idx+2])
        cumulative = np.zeros_like(sorted_idx, dtype=np.float64)
        current_p = 0
        credible_level = (0.05, 0.16, 0.5, 0.84, 0.95)
        credible_point = np.zeros(len(credible_level), dtype=np.int)
        cur = 0
        for i in range(0, sorted_idx.shape[0]):
            current_p += self.ftxt[sorted_idx[i], 0]
            if cur < len(credible_level) and current_p >= credible_level[cur]:
                credible_point[cur] = i
                cur += 1
            cumulative[i] = current_p
        if color is not None and label is not None:
            ax.plot(self.ftxt[sorted_idx, idx+2], cumulative, color=color, label=label)
        elif color is not None:
            ax.plot(self.ftxt[sorted_idx, idx+2], cumulative, color=color)
        elif label is not None:
            ax.plot(self.ftxt[sorted_idx, idx+2], cumulative, label=label)
        else:
            ax.plot(self.ftxt[sorted_idx, idx + 2], cumulative)
        if color is not None:
            ax.fill_between(self.ftxt[sorted_idx[credible_point[0]: credible_point[4]+1], idx+2], 0,
                            cumulative[credible_point[0]: credible_point[4]+1], color=color, alpha=0.2)
            ax.fill_between(self.ftxt[sorted_idx[credible_point[1]: credible_point[3] + 1], idx + 2], 0,
                            cumulative[credible_point[1]: credible_point[3] + 1], color=color, alpha=0.3)
        else:
            ax.fill_between(self.ftxt[sorted_idx[credible_point[0]: credible_point[4] + 1], idx + 2], 0,
                            cumulative[credible_point[0]: credible_point[4] + 1], alpha=0.2)
            ax.fill_between(self.ftxt[sorted_idx[credible_point[1]: credible_point[3] + 1], idx + 2], 0,
                            cumulative[credible_point[1]: credible_point[3] + 1], alpha=0.3)
        for vert in credible_point[2:3]:
            if color is not None:
                ax.axvline(x=self.ftxt[sorted_idx[vert], idx+2], ymax=cumulative[vert]*0.95, linestyle='--', color=color)
            else:
                ax.axvline(x=self.ftxt[sorted_idx[vert], idx + 2], ymax=cumulative[vert] * 0.95, linestyle='--')
        ax.set_ylim(0, 1)
        return fig, ax

    # def credible_2d_equal_weights(self, idx: tuple, credible_level=(0.6827, 0.9545), nbins=80, ax=None, center=None, heat=False):
    #     """
    #     plot the correlation between parameters
    #     :param idx: the index of the two parameters to be ploted
    #     :param credible_level: choose which credible levels to plot
    #     :param nbins: number of bins
    #     :param ax: axes to be plot on, if not none
    #     :param center: mark center point
    #     :return: figure and axes object for further fine tuning the plot
    #     """
    #     if ax is not None:
    #         fig = None
    #     else:
    #         fig, ax = subplots()
    #     minx = np.amin(self.ftxt[:, idx[0]])
    #     miny = np.amin(self.ftxt[:, idx[1]])
    #     maxx = np.amax(self.ftxt[:, idx[0]])
    #     maxy = np.amax(self.ftxt[:, idx[1]])
    #     binxw = (maxx - minx) / nbins
    #     binyw = (maxy - miny) / nbins
    #     binx = np.linspace(minx + binxw/2, maxx - binxw/2, nbins)
    #     biny = np.linspace(miny + binyw/2, maxy - binyw/2, nbins)
    #     xv, yv = np.meshgrid(binx, biny)
    #     zv = np.zeros_like(xv)
    #     # be careful that position in x direction is column, position in y direction is row!
    #     for i in range(self.ftxt.shape[0]):
    #         posx = int((self.ftxt[i, idx[0]] - minx) / binxw)
    #         posy = int((self.ftxt[i, idx[1]] - miny) / binyw)
    #         if posx < nbins and posy < nbins:
    #             zv[posy, posx] += 1
    #         elif posy < nbins:
    #             zv[posy, posx-1] += 1
    #         elif posx < nbins:
    #             zv[posy-1, posx] += 1
    #         else:
    #             zv[posy-1, posx-1] += 1
    #     zv = zv / self.ftxt.shape[0]
    #     sorted_idx = np.unravel_index(
    #         np.argsort(zv, axis=None)[::-1], zv.shape)
    #     if heat:
    #         im = ax.pcolormesh(xv, yv, zv/(binxw*binyw),
    #                            cmap='rainbow', edgecolors='face')
    #         # fig.colorbar(im)
    #         # fig.colorbar()
    #     else:
    #         cl = np.sort(credible_level)[::-1]
    #         al = np.linspace(0.2, 0.3, len(cl))
    #         cll = 0
    #         for ic in range(len(cl)):
    #             cz = np.zeros_like(zv)
    #             s = 0
    #             for i in range(sorted_idx[0].shape[0]):
    #                 s += zv[sorted_idx[0][i], sorted_idx[1][i]]
    #                 cz[sorted_idx[0][i], sorted_idx[1][i]
    #                    ] = zv[sorted_idx[0][i], sorted_idx[1][i]]
    #                 if s > cl[ic]:
    #                     cll = zv[sorted_idx[0][i], sorted_idx[1][i]]
    #                     break
    #             ax.contourf(xv, yv, zv, (cll, 1), colors=(
    #                 'b', 'white'), alpha=al[ic])
    #         ax.axis('scaled')
    #     if center is not None:
    #         ax.plot(np.array([center[0]]), np.array([center[1]]), "*")
    #     xleft, xright = ax.get_xlim()
    #     ybottom, ytop = ax.get_ylim()
    #     ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)))
    #     return fig, ax
