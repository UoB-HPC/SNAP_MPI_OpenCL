SUBROUTINE population ( pop )

!_______________________________________________________________________
!
! Sum the scalar flux, weighted by cell volume to calculate
! particle population
!_______________________________________________________________________

  USE global_module, ONLY: i_knd, r_knd

  USE solvar_module, ONLY: flux

  USE geom_module, ONLY: dx, dy, dz

  USE data_module, ONLY: ng

  USE plib_module, ONLY: comm_snap, root


  IMPLICIT NONE

  INCLUDE 'mpif.h'


  REAL(r_knd), INTENT(OUT) :: pop


!_______________________________________________________________________
!
! Local variables
!_______________________________________________________________________

  REAL(r_knd) :: volume, total = 0.0_r_knd
  INTEGER(i_knd) :: ierr

  volume = dx * dy * dz

  total = SUM(flux(:,:,:,:) * volume)

  IF ( comm_snap == MPI_COMM_NULL ) THEN
    pop = total
    RETURN
  ENDIF

  CALL MPI_REDUCE ( total, pop, 1, MPI_DOUBLE_PRECISION, MPI_SUM, root, comm_snap, ierr )

END SUBROUTINE population
