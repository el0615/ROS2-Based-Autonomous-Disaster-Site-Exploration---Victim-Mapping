/* DN-011 visibility DDA accelerator. Only rasterizes hypothetical ray visibility.
 * It does not alter the map, coverage, costmap, footprint, or navigation controls.
 * Built from source on the robot computer; no third-party C dependencies.
 */
#include <math.h>
#include <stdint.h>
#include <stddef.h>

static inline int free_at(const uint8_t *free_grid, int width, int height, int x, int y) {
    return x >= 0 && y >= 0 && x < width && y < height &&
           free_grid[(size_t)y * (size_t)width + (size_t)x] != 0;
}

/* All input arrays length=beams, output 'seen' length=width*height, zero-initialized.
   Each beam follows the same cell-entry, grid-corner, and range comparisons as
   observation_planning_core.visibility_cells. No unsafe rays are skipped.
*/
void dn011_visibility_dda(const uint8_t *free_grid, int width, int height,
                          const double *dirx, const double *diry,
                          const double *startx, const double *starty,
                          int beams, double limit, uint8_t *seen) {
    const double inf = INFINITY;
    for (int beam = 0; beam < beams; ++beam) {
        double dx = dirx[beam], dy = diry[beam];
        double px = startx[beam], py = starty[beam];
        int ix = (int)floor(px), iy = (int)floor(py);
        int stepx = dx > 0.0 ? 1 : -1;
        int stepy = dy > 0.0 ? 1 : -1;
        double txdelta = fabs(dx) > 1e-12 ? 1.0 / fabs(dx) : inf;
        double tydelta = fabs(dy) > 1e-12 ? 1.0 / fabs(dy) : inf;
        double fx = dx > 0.0 ? (ix + 1.0 - px) : (px - ix);
        double fy = dy > 0.0 ? (iy + 1.0 - py) : (py - iy);
        double tx = isfinite(txdelta) ? fx * txdelta : inf;
        double ty = isfinite(tydelta) ? fy * tydelta : inf;
        double travelled = 0.0;
        for (int iteration = 0; iteration < width + height + 4; ++iteration) {
            if (!free_at(free_grid, width, height, ix, iy)) break;
            double end = fmin(fmin(tx, ty), limit);
            if (end > travelled + 1e-10) {
                seen[(size_t)iy * (size_t)width + (size_t)ix] = 1;
            }
            if (end >= limit - 1e-10) break;
            if (fabs(tx - ty) < 1e-10) {
                if (!free_at(free_grid, width, height, ix + stepx, iy) ||
                    !free_at(free_grid, width, height, ix, iy + stepy)) break;
                travelled = tx;
                ix += stepx;
                iy += stepy;
                tx += txdelta;
                ty += tydelta;
            } else if (tx < ty) {
                travelled = tx;
                ix += stepx;
                tx += txdelta;
            } else {
                travelled = ty;
                iy += stepy;
                ty += tydelta;
            }
        }
    }
}
