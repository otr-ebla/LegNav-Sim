import os

# === FORZA JAX A CUDA PRIMA DI IMPORTARE JAX ===
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["JAX_PLATFORM_NAME"] = "cuda"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/LABAUT/alberto_vaglio/cuda12-local"

import jax
import jax.numpy as jnp
from jax import lax

@jax.jit    
def bounce_vs_circles(people_pos, people_vel,
                      circle_centers, circle_radii,
                      people_radius, restitution=0.0):
    """
    people_pos:   (P, 2)
    people_vel:   (P, 2)
    circle_centers: (C, 2)
    circle_radii:   (C,)
    """
    if circle_centers is None or circle_centers.shape[0] == 0:
        return people_pos, people_vel

    # (P, C, 2)
    diff = people_pos[:, None, :] - circle_centers[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)              # (P, C)

    min_dist = people_radius + circle_radii[None, :]   # (P, C)
    penetration = jnp.maximum(0.0, min_dist - dist)    # (P, C)

    # Normali (P, C, 2)
    normal = diff / (dist[..., None] + 1e-8)

    pos_corr_each = normal * penetration[..., None]    # (P, C, 2)

    # sommo tutte le correzioni da tutti i cerchi
    pos_corr_total = jnp.sum(pos_corr_each, axis=1)    # (P, 2)
    people_pos = people_pos + pos_corr_total

    collided = jnp.any(penetration > 0.0, axis=1)  # (P,)

    # se hai toccato qualcosa, semplicemente inverti la velocità
    people_vel = jnp.where(collided[:, None], -people_vel, people_vel)

    return people_pos, people_vel

@jax.jit
def bounce_vs_rectangles(people_pos, people_vel,
                         rect_centers, rect_sizes,
                         people_radius, restitution=0.0):
    """
    rect_centers: (R, 2)
    rect_sizes:  (R, 2) full sizes (width, height)
    people_pos:  (P, 2)
    """
    if rect_centers is None or rect_centers.shape[0] == 0:
        return people_pos, people_vel

    half = rect_sizes / 2.0                              # (R, 2)

    # (P, R, 2)
    diff = people_pos[:, None, :] - rect_centers[None, :, :]

    # punto più vicino dentro il rettangolo
    clamped = jnp.clip(diff, -half[None, :, :], half[None, :, :])
    closest = rect_centers[None, :, :] + clamped        # (P, R, 2)

    # vettore dal closest al centro della persona
    d = people_pos[:, None, :] - closest                # (P, R, 2)
    dist = jnp.linalg.norm(d, axis=-1)                  # (P, R)

    penetration = jnp.maximum(0.0, people_radius - dist) # (P, R)

    # normali
    normal = d / (dist[..., None] + 1e-8)               # (P, R, 2)

    # --- correzione posizione ---
    pos_corr_each = normal * penetration[..., None]     # (P, R, 2)
    pos_corr_total = jnp.sum(pos_corr_each, axis=1)     # (P, 2)
    people_pos = people_pos + pos_corr_total

    # --- riflessione velocità ---
    collided = jnp.any(penetration > 0.0, axis=1)  # (P,)
    people_vel = jnp.where(collided[:, None], -people_vel, people_vel)

    return people_pos, people_vel


@jax.jit
def bounce_people_on_obstacles(people_pos, people_vel,
                               circles, rectangles,
                               cfg, restitution=0.0, max_iterations=4):
    """
    Applica più iterazioni di risoluzione delle collisioni
    per essere sicuro di eliminare la penetrazione.
    """

    circle_centers = None
    circle_radii = None
    if (circles is not None) and (circles.centers.size > 0):
        circle_centers = circles.centers  # (C, 2)
        circle_radii   = circles.radii    # (C,)


    rect_centers = None
    rect_sizes   = None
    if (rectangles is not None) and (rectangles.x_min.size > 0):
        xmin = rectangles.x_min
        ymin = rectangles.y_min
        xmax = rectangles.x_max
        ymax = rectangles.y_max

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        rect_centers = jnp.stack([cx, cy], axis=-1)  #

        w = xmax - xmin
        h = ymax - ymin
        rect_sizes = jnp.stack([w, h], axis=-1)      #

    def body_fun(_, carry):
        p, v = carry

        # 1) rimbalzo contro cerchi
        if circle_centers is not None:
            p, v = bounce_vs_circles(p, v,
                                     circle_centers, circle_radii,
                                     cfg.people_radius, restitution)

        # 2) rimbalzo contro rettangoli
        if rect_centers is not None:
            p, v = bounce_vs_rectangles(p, v,
                                        rect_centers, rect_sizes,
                                        cfg.people_radius, restitution)

        # 3) rimbalzo tra persone
        p, v = bounce_vs_people(p, v,
                                cfg.people_radius, restitution)
        
        p, v = bounce_vs_walls(p, v,
                               cfg.people_radius,
                               cfg.room_width,
                               cfg.room_height,
                               restitution)

        return (p, v)

    # iterazioni (pochissime, perché Step 1 già riduce la penetrazione)
    p_final, v_final = jax.lax.fori_loop(
        0, max_iterations, body_fun, (people_pos, people_vel)
    )

    return p_final, v_final


@jax.jit
def bounce_vs_people(people_pos, people_vel, people_radius, restitution=0.0):
    """ Gestisce le collisioni e il rimbalzo tra le persone. """
    P = people_pos.shape[0]
    
    # Non eseguire se ci sono meno di 2 persone
    # (Usiamo jnp.where per il controllo condizionale in un contesto JIT)
    if P <= 1:
        return people_pos, people_vel

    # (P, P, 2) - Vettore differenza tra tutte le coppie (diff[i, j] = pos[i] - pos[j])
    diff = people_pos[:, None, :] - people_pos[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)  # (P, P)

    # Maschera per escludere P vs P (un oggetto non collide con se stesso)
    i = jnp.arange(P)
    mask = i[:, None] != i[None, :]

    # Distanza minima di non compenetrazione (2*r)
    min_dist = 2.0 * people_radius
    
    # Penetrazione: max(0, min_dist - dist). Mascherata per escludere P vs P.
    penetration_unmasked = jnp.maximum(0.0, min_dist - dist)
    penetration = jnp.where(mask, penetration_unmasked, 0.0) # (P, P)

    # Normali (P, P, 2)
    # n = diff / dist (vettore normalizzato dalla persona 'j' alla persona 'i')
    normal = diff / (dist[..., None] + 1e-8)

    # --- A) Correzione di Posizione ---
    # Dividiamo per 2 perché la penetrazione viene calcolata da entrambe le parti
    pos_corr_each = normal * penetration[..., None] / 2.0
    
    # Somma delle correzioni subite da 'i' da parte di tutti gli 'j'
    pos_corr_total = jnp.sum(pos_corr_each, axis=1)    # (P, 2)
    people_pos = people_pos + pos_corr_total

    # --- B) Riflessione Velocità (Inversione semplificata) ---
    # Una persona 'i' ha subito collisione se c'è penetrazione con almeno un 'j'
    collided = jnp.any(penetration > 0.0, axis=1)  # (P,)
    
    # Se hai toccato qualcosa, inverti la velocità
    people_vel = jnp.where(collided[:, None], -people_vel, people_vel)

    return people_pos, people_vel

# Aggiungi questa funzione nel tuo file bouncing.py

@jax.jit
def bounce_vs_walls(people_pos, people_vel, people_radius, room_width, room_height, restitution=0.0):
    """ Gestisce le collisioni e il rimbalzo delle persone contro le pareti. """
    
    # 1. Correzione Posizione (per rimuovere la penetrazione)
    
    # Calcola la penetrazione massima in ciascuna direzione
    pen_left = jnp.maximum(0.0, people_radius - people_pos[:, 0])
    pen_right = jnp.maximum(0.0, people_radius - (room_width - people_pos[:, 0]))
    pen_bottom = jnp.maximum(0.0, people_radius - people_pos[:, 1])
    pen_top = jnp.maximum(0.0, people_radius - (room_height - people_pos[:, 1]))
    
    # Aggiorna la posizione
    people_pos = people_pos.at[:, 0].add(pen_left - pen_right)
    people_pos = people_pos.at[:, 1].add(pen_bottom - pen_top)
    
    # 2. Riflessione Velocità
    
    # Maschera per le collisioni: True se c'è stata penetrazione (che ora è corretta)
    collided_x = (pen_left > 0.0) | (pen_right > 0.0)
    collided_y = (pen_bottom > 0.0) | (pen_top > 0.0)
    
    # Rifletti la componente X se c'è collisione orizzontale
    people_vel = people_vel.at[:, 0].set(
        jnp.where(collided_x, -people_vel[:, 0], people_vel[:, 0])
    )
    # Rifletti la componente Y se c'è collisione verticale
    people_vel = people_vel.at[:, 1].set(
        jnp.where(collided_y, -people_vel[:, 1], people_vel[:, 1])
    )
    
    return people_pos, people_vel