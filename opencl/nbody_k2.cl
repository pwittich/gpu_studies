/* nbody_kern.cl  -*-c-*- */
/* (c) Copyright © 2009-2010 Brown Deer Technology, LLC */

__kernel void nbody_k2(
		       float dt1, float eps, int nsteps, 
		       __global float4* pos_io,
		       __global float4* vel_io,
		       __local float4* pos0,
		       __local float4* pos1
		       ) 
{
  const float4 dt = (float4){dt1,dt1,dt1,0.0f};

  int gti = get_global_id(0);
  int ti = get_local_id(0);

  int n = get_global_size(0);
  int nt = get_local_size(0);
  int nb = n/nt;

  event_t copydone[2];

  copydone[0] = async_work_group_copy(pos1, pos_io,
				      n, (event_t)0);

  // copydone[1] = async_work_group_copy(vel, vel_io,
  // 				      n, (event_t)0); // do I need this 

  wait_group_events(1, copydone);


  for (int istep = 0; istep < nsteps; ++istep ) { // loop over steps
    __local float4 *pos_start = pos1;
    __local float4 *pos_end   = pos0;
    if ( istep % 2 == 1 ) {
      pos_start = pos0;
      pos_end   = pos1;
    }

    float4 p = pos_start[gti];
    float4 v = vel_io[gti];
    float4 a = (float4)(0.0f,0.0f,0.0f,0.0);

    for ( int i = 0; i < n; ++i ) { // loop over particles acting on me
      if ( i == gti ) continue; // no self-interactions
      float4 p2 = pos_start[i]; /* Read a cached particle position */

      float4 d = p2 - p;
      float invr = rsqrt(d.x*d.x + d.y*d.y + d.z*d.z + eps );
      float f = p2.w*invr*invr*invr; // this is actually f/(r^3 m_1), assume G = 1
      //float f = p2.w*invr;
      // d is direction btw two but not a unit vector
      // extra powers of invr above take care of length
      a += f*d; /* Accumulate acceleration */

    }

    p += dt*v + 0.5f*dt*dt*a;
    v += dt*a;

    pos_end[gti] = p;
    vel_io[gti] = v;
    barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in work-group */

  }

  copydone[0] = async_work_group_copy(pos_io, pos1,
				      n, (event_t)0);

  // copydone[1] = async_work_group_copy(vel_io, vel,
  // 				      n, (event_t)0); // do I need this 

  wait_group_events(1, &copydone);



}

