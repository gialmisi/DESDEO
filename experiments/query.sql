WITH RECURSIVE tree AS (
  -- roots in this session (often one, but can be >1)
  SELECT
    id            AS node_id,
    parent_id     AS parent_node_id,
    state_id      AS state_id,
    problem_id    AS problem_id,
    session_id    AS session_id,
    0             AS depth
  FROM statedb
  WHERE session_id = {SESSION_ID} AND parent_id IS NULL

  UNION ALL

  SELECT
    s.id          AS node_id,
    s.parent_id   AS parent_node_id,
    s.state_id    AS state_id,
    s.problem_id  AS problem_id,
    s.session_id  AS session_id,
    t.depth + 1   AS depth
  FROM statedb s
  JOIN tree t ON s.parent_id = t.node_id
)
SELECT
  t.*,
  st.method,
  st.phase,
  st.kind,

  -- session metadata (optional but usually useful)
  sess.user_id   AS session_user_id,
  sess.info      AS session_info,

  -- E-NAUTILUS payload (nullable for non-E-NAUTILUS nodes)
  e.current_iteration,
  e.iterations_left,
  e.selected_point,
  e.reachable_point_indices,
  e.number_of_intermediate_points,
  e.enautilus_results

FROM tree t
JOIN states st ON st.id = t.state_id
LEFT JOIN interactivesessiondb sess ON sess.id = t.session_id
LEFT JOIN enautilusstate e ON e.id = t.state_id
ORDER BY t.depth, t.node_id;