import os
import logging
import numpy as np
import geopandas as gpd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.contrib.gis.geos import GEOSGeometry, Polygon, MultiPolygon
from base.models import SpatialGridCell

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Bulk seed $N=22,584$ spatial grid cells and socio-demographic metrics into PostGIS."

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            '-f',
            type=str,
            default=None,
            help='Path to spatial file (GeoJSON, Shapefile, or GeoPackage) containing grid geometries.'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=2000,
            help='PostgreSQL bulk insert transaction chunk size (default: 2000).'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing SpatialGridCell records before seeding.'
        )
        parser.add_argument(
            '--synthetic',
            action='store_true',
            help='Generate synthetic $N=22,584$ 1km² grid cells over target AOI if vector file is absent.'
        )

    def handle(self, *args, **options):
        file_path = options['file']
        batch_size = options['batch_size']
        should_clear = options['clear']
        use_synthetic = options['synthetic']

        self.stdout.write(self.style.MIGRATE_HEADING("=== Starting GeoAfya Spatial Grid Seeding ==="))

        # 1. Clear Existing Cells if requested
        if should_clear:
            self.stdout.write(self.style.WARNING("Clearing existing SpatialGridCell records from database..."))
            with transaction.atomic():
                count, _ = SpatialGridCell.objects.all().delete()
            self.stdout.write(self.style.SUCCESS(f"Deleted {count} existing grid cell records."))

        # 2. Read or Generate GeoDataFrame
        if file_path and os.path.exists(file_path):
            self.stdout.write(f"Loading spatial geometries from file: {file_path}")
            gdf = self._load_from_file(file_path)
        elif use_synthetic or not file_path:
            self.stdout.write(self.style.WARNING(
                "No spatial file specified or found. Generating synthetic N=22,584 (151x150) grid cells over target AOI..."
            ))
            gdf = self._generate_synthetic_grid(n_cells=22584)
        else:
            raise CommandError(f"Specified vector file does not exist: {file_path}")

        total_cells = len(gdf)
        self.stdout.write(self.style.SUCCESS(f"Loaded {total_cells} grid cell geometries. Preparing PostGIS objects..."))

        # 3. Build Model Instances
        objects_to_create = []
        for idx, row in gdf.iterrows():
            cell_id = int(row.get('cell_id', idx + 1))
            
            # Convert geometry to Django GEOS Geometry & ensure EPSG:4326 MultiPolygon
            geom_json = row['geometry'].__geo_interface__
            geos_geom = GEOSGeometry(str(geom_json))
            
            if isinstance(geos_geom, Polygon):
                geos_geom = MultiPolygon(geos_geom)
            elif not isinstance(geos_geom, MultiPolygon):
                self.stdout.write(self.style.WARNING(f"Skipping cell {cell_id}: Invalid geometry type ({type(geos_geom)})."))
                continue

            geos_geom.srid = 4326

            # Extract or assign default socio-demographic features
            poverty_rate = float(row.get('poverty_rate', np.random.uniform(10.0, 65.0)))
            malnutrition_rate = float(row.get('malnutrition_rate', np.random.uniform(2.0, 35.0)))
            health_travel_time = float(row.get('health_travel_time', np.random.uniform(10.0, 180.0)))
            pop_density = float(row.get('pop_density', np.random.uniform(5.0, 850.0)))
            settlement_dist = float(row.get('settlement_dist', np.random.uniform(0.5, 30.0)))
            healthcare_deficit = float(row.get('healthcare_deficit', np.random.uniform(0.1, 0.9)))

            grid_cell = SpatialGridCell(
                cell_id=cell_id,
                geom=geos_geom,
                poverty_rate=poverty_rate,
                malnutrition_rate=malnutrition_rate,
                health_travel_time=health_travel_time,
                pop_density=pop_density,
                settlement_dist=settlement_dist,
                healthcare_deficit=healthcare_deficit
            )
            objects_to_create.append(grid_cell)

        # 4. Perform Bulk Persistence
        self.stdout.write(f"Executing bulk PostGIS inserts in chunks of {batch_size}...")
        created_count = 0
        
        with transaction.atomic():
            for i in range(0, len(objects_to_create), batch_size):
                chunk = objects_to_create[i:i + batch_size]
                SpatialGridCell.objects.bulk_create(chunk, batch_size=batch_size, ignore_conflicts=True)
                created_count += len(chunk)
                self.stdout.write(f"  --> Progress: {created_count}/{total_cells} cells seeded...")

        self.stdout.write(self.style.SUCCESS(
            f" Seeding Complete! Successfully inserted {created_count} spatial grid cells into base_spatialgridcell."
        ))

    def _load_from_file(self, file_path: str) -> gpd.GeoDataFrame:
        """Loads spatial vector data and reprojects to WGS84 (EPSG:4326)."""
        gdf = gpd.read_file(file_path)
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            self.stdout.write("Reprojecting spatial layer CRS to EPSG:4326...")
            gdf = gdf.to_crs(epsg=4326)
        return gdf

    def _generate_synthetic_grid(self, n_cells: int = 22584) -> gpd.GeoDataFrame:
        """
        Generates a 1km² regular mesh grid bounding box (~151x150 cells = 22,650 cells)
        over target East African epidemiological AOI bounds [36.0, -1.5, 37.35, -0.15].
        """
        # Define AOI extent (Bounding box around East Africa target zone)
        min_x, min_y = 36.00, -1.50
        max_x, max_y = 37.35, -0.15

        # 1km approx in degrees at equator = 0.008983 degrees
        cell_size = 0.008983
        
        cols = int(np.ceil((max_x - min_x) / cell_size))
        rows = int(np.ceil((max_y - min_y) / cell_size))

        polygons = []
        cell_ids = []
        count = 1

        for i in range(cols):
            for j in range(rows):
                if count > n_cells:
                    break
                
                x1 = min_x + (i * cell_size)
                y1 = min_y + (j * cell_size)
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                poly = shapely_polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])
                polygons.append(poly)
                cell_ids.append(count)
                count += 1
            if count > n_cells:
                break

        # Generate realistic synthetic baseline distributions
        np.random.seed(42)
        gdf = gpd.GeoDataFrame({
            'cell_id': cell_ids,
            'poverty_rate': np.random.beta(2, 5, size=len(cell_ids)) * 100.0,
            'malnutrition_rate': np.random.beta(2, 6, size=len(cell_ids)) * 50.0,
            'health_travel_time': np.random.gamma(2, 20, size=len(cell_ids)),
            'pop_density': np.random.exponential(150, size=len(cell_ids)),
            'settlement_dist': np.random.exponential(5, size=len(cell_ids)),
            'healthcare_deficit': np.random.uniform(0.1, 0.9, size=len(cell_ids)),
            'geometry': polygons
        }, crs="EPSG:4326")

        return gdf


def shapely_polygon(coords):
    from shapely.geometry import Polygon as ShapelyPoly
    return ShapelyPoly(coords)