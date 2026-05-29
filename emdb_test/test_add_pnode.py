#!/usr/bin/env python3
"""
Script de dreaming para e-MDB.
Inyecta puntos (positivos) y anti-puntos (negativos) en un PNode existente
usando /pnode/<name>/add_points con la firma real:
    request.points[]       → lista de Perception
    request.confidences[]  → lista de floats (+1.0 = punto, -1.0 = anti-punto)
"""

import rclpy
from rclpy.node import Node
import numpy as np
import sys
import yaml

# ---------------------------------------------------------------------------
# Espacio perceptual completo (del point_msg leído del PNode real pick_fruit)
# ---------------------------------------------------------------------------
PERCEPTION_DIMS = [
    ("button_light",        0.0, 1.0),
    ("fruit_in_left_hand",  0.0, 1.0),
    ("fruit_in_right_hand", 0.0, 1.0),
    ("fruits.distance",     0.0, 1.0),
    ("fruits.angle",        0.0, 1.0),
    ("fruits.dim_max",      0.0, 1.0),
    ("scales.distance",     0.0, 1.0),
    ("scales.angle",        0.0, 1.0),
    ("scales.state",        0.0, 1.0),
    ("scales.active",       0.0, 1.0),
]

# ---------------------------------------------------------------------------
# Región POSITIVA: condiciones donde pick_fruit DEBERÍA funcionar
# Robot sin fruta en manos + fruta cerca → ejecutar pick tiene sentido
# ---------------------------------------------------------------------------
POSITIVE_REGION = {
    "fruit_in_left_hand":  (0.0, 0.05),   # sin fruta en mano izq
    "fruit_in_right_hand": (0.0, 0.05),   # sin fruta en mano der
    "fruits.distance":     (0.25, 0.15),  # fruta cerca
    "scales.active":       (0.0, 0.05),   # balanza inactiva
}

# ---------------------------------------------------------------------------
# Región NEGATIVA (anti-puntos): condiciones donde pick_fruit NO funciona
# Robot ya tiene fruta en mano, o no hay fruta cerca
# ---------------------------------------------------------------------------
NEGATIVE_REGION = {
    "fruit_in_left_hand":  (1.0, 0.05),   # ya tiene fruta en mano izq
    "fruits.distance":     (0.85, 0.10),  # fruta muy lejos
}

N_POSITIVE  = 200   # puntos positivos a inyectar
N_NEGATIVE  = 100   # anti-puntos a inyectar
BATCH_SIZE  = 50    # tamaño de lote
RANDOM_SEED = 42
DEFAULT_PNODE_CLASS = 'cognitive_nodes.pnode.PNode'
DEFAULT_SPACE_CLASS = 'cognitive_nodes.space.PointBasedSpace'


def _default_schema_from_dims():
    schema = []
    grouped = {}
    for index, (dim_name, _, _) in enumerate(PERCEPTION_DIMS):
        if '.' in dim_name:
            sensor, attribute = dim_name.split('.', 1)
            region_key = dim_name
        else:
            sensor, attribute = dim_name, 'data'
            region_key = sensor
        grouped.setdefault((index, sensor), []).append((attribute, region_key))

    for (index, sensor), attributes in grouped.items():
        schema.append({'index': index, 'sensor': sensor, 'attributes': attributes})
    return schema


def _schema_from_labels(labels):
    schema = []
    grouped = {}
    for label in labels:
        try:
            index_str, sensor, attribute = label.split('-', 2)
            index = int(index_str)
        except ValueError:
            index = len(grouped)
            sensor = label
            attribute = 'data'

        region_key = sensor if attribute == 'data' else f'{sensor}.{attribute}'
        grouped.setdefault((index, sensor), []).append((attribute, region_key))

    for (index, sensor), attributes in sorted(grouped.items(), key=lambda item: item[0][0]):
        schema.append({'index': index, 'sensor': sensor, 'attributes': attributes})
    return schema or _default_schema_from_dims()


def generate_points(n, schema, region, seed):
    """Genera n puntos muestreados en la región dada."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n):
        pt = {}
        for item in schema:
            sensor = item['sensor']
            values = {}
            for attribute, region_key in item['attributes']:
                lo, hi = 0.0, 1.0
                if region_key in region:
                    center, margin = region[region_key]
                    val = rng.uniform(max(lo, center - margin), min(hi, center + margin))
                else:
                    val = rng.uniform(lo, hi)
                values[attribute] = float(val)
            pt.setdefault(sensor, []).append(values)
        points.append(pt)
    return points


class DreamingInjector(Node):
    def __init__(self, pnode_name):
        super().__init__('dreaming_injector')
        self.pnode_name = pnode_name

        from cognitive_node_interfaces.srv import AddPoints
        from cognitive_node_interfaces.srv import SendSpace
        from core_interfaces.srv import CreateNode
        self.AddPoints = AddPoints
        self.SendSpace = SendSpace
        self.CreateNode = CreateNode

        self.add_points_service = f'/pnode/{pnode_name}/add_points'
        self.send_space_service = f'/pnode/{pnode_name}/send_space'
        self.cli = self.create_client(AddPoints, self.add_points_service)
        self.space_cli = self.create_client(SendSpace, self.send_space_service)
        self.create_cli = self.create_client(CreateNode, 'commander/create')
        self.schema = _default_schema_from_dims()
        self.ensure_pnode()

    def ensure_pnode(self):
        """Garantiza que exista el PNode y su servicio add_points."""
        self.get_logger().info(f'Esperando servicio {self.add_points_service}...')
        if self.cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('Servicio disponible.')
            self._load_schema_from_space()
            return

        self.get_logger().warn(
            f'Servicio {self.add_points_service} no disponible. Intentando crear PNode...'
        )

        if not self.create_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Servicio commander/create no disponible.')
            sys.exit(1)

        req = self.CreateNode.Request()
        req.name = self.pnode_name
        req.class_name = DEFAULT_PNODE_CLASS
        req.parameters = yaml.dump(
            {
                'space_class': DEFAULT_SPACE_CLASS,
                'history_size': 100,
            },
            sort_keys=False,
        )

        future = self.create_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)
        result = future.result()
        if result is None:
            self.get_logger().error('No se recibió respuesta de commander/create.')
            sys.exit(1)

        if result.created:
            self.get_logger().info(f'PNode {self.pnode_name} creado correctamente.')
        else:
            self.get_logger().warn(
                f'commander/create devolvió created=False para {self.pnode_name}. '
                'Puede que ya exista o que hubiera un error previo.'
            )

        self.get_logger().info(f'Esperando de nuevo {self.add_points_service}...')
        if not self.cli.wait_for_service(timeout_sec=20.0):
            self.get_logger().error(f'Servicio {self.add_points_service} no disponible tras crear el PNode.')
            sys.exit(1)

        self.get_logger().info('Servicio disponible.')
        self._load_schema_from_space()

    def _load_schema_from_space(self):
        """Recupera las labels reales del espacio para construir percepciones compatibles."""
        if not self.space_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f'Servicio {self.send_space_service} no disponible; usando esquema por defecto.')
            self.schema = _default_schema_from_dims()
            return

        future = self.space_cli.call_async(self.SendSpace.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result()
        if response is None or not getattr(response, 'labels', None):
            self.get_logger().warn('No se pudieron leer labels del PNode; usando esquema por defecto.')
            self.schema = _default_schema_from_dims()
            return

        self.schema = _schema_from_labels(response.labels)
        self.get_logger().info(f'Esquema de percepción cargado con {len(self.schema)} objetos.')

    def _send_batch(self, points, confidence):
        """Envía un lote de puntos con la confianza dada (+1.0 o -1.0)."""
        from cognitive_node_interfaces.msg import Perception, ObjectParameters

        def point_to_perception_msg(point_values):
            # Build a Perception message compatible with core.utils.msg_to_dict.
            msg = Perception()
            msg.layout.data_offset = 0
            len_float = 8

            for sensor, instances in point_values.items():
                for obj_index, attributes in enumerate(instances):
                    dim = ObjectParameters()
                    dim.size_stride_units = 'bytes'
                    dim.object = f'{sensor}{obj_index}'
                    dim.labels = list(attributes.keys())
                    dim.size = len_float * len(attributes)
                    dim.stride = len_float
                    msg.layout.dim.append(dim)

                    for attribute, value in attributes.items():
                        msg.data.append(float(value))
                        msg.is_valid.append(True)

            return msg

        req = self.AddPoints.Request()
        perceptions = []
        for pt in points:
            p = point_to_perception_msg(pt)
            perceptions.append(p)
        req.points = perceptions
        req.confidences = [float(confidence)] * len(points)

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if future.exception() is not None:
            self.get_logger().error(f'Excepción en add_points: {future.exception()}')
            return False
        result = future.result()
        return result is not None and result.added

    def inject(self, points, confidence, label):
        """Inyecta una lista de puntos con la confianza dada, en lotes."""
        total = len(points)
        injected = 0
        sign = '+' if confidence > 0 else '-'
        self.get_logger().info(
            f'[{label}] Inyectando {total} puntos (confianza={sign}{abs(confidence)})...'
        )
        for start in range(0, total, BATCH_SIZE):
            batch = points[start:start + BATCH_SIZE]
            ok = self._send_batch(batch, confidence)
            if ok:
                injected += len(batch)
                self.get_logger().info(
                    f'  Lote {start//BATCH_SIZE + 1}: {injected}/{total}'
                )
            else:
                self.get_logger().error(
                    f'  Lote {start//BATCH_SIZE + 1}: fallo.'
                )
        self.get_logger().info(
            f'[{label}] Completado: {injected}/{total} puntos inyectados.\n'
        )
        return injected


def main():
    pnode_name = (sys.argv[1] if len(sys.argv) > 1
                  else 'pnode_FRUIT_SHOP__effect_fruit_in_left_hand_data__pick_fruit')

    print(f'\n[Dreaming] PNode objetivo : {pnode_name}')
    print(f'[Dreaming] Puntos positivos: {N_POSITIVE}  |  Anti-puntos: {N_NEGATIVE}\n')

    rclpy.init()
    node = DreamingInjector(pnode_name)

    # 1. Puntos positivos (confidence = +1.0)
    positive_pts = generate_points(N_POSITIVE, node.schema, POSITIVE_REGION, seed=RANDOM_SEED)
    node.inject(positive_pts, confidence=1.0, label='POSITIVOS')

    # 2. Anti-puntos (confidence = -1.0)
    negative_pts = generate_points(N_NEGATIVE, node.schema, NEGATIVE_REGION, seed=RANDOM_SEED + 1)
    node.inject(negative_pts, confidence=-1.0, label='ANTI-PUNTOS')

    node.destroy_node()
    rclpy.shutdown()
    print('[Dreaming] Finalizado.')


if __name__ == '__main__':
    main()
